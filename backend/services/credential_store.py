"""
Credential Store
=================
Secure in-memory storage for exchange API credentials.

Security model:
    - Keys are encrypted with AES-256-GCM before storage
    - Encryption key is derived from a server-side secret (never sent to client)
    - Raw keys exist in memory only during connect() validation
    - Frontend never receives keys back after submission
    - Session tokens are random UUIDs with no key material embedded
    - Keys are stored per session_id — multi-user safe
    - No keys are ever written to disk or logs

In production: replace _store dict with Redis + TTL for persistence
across server restarts and multi-instance deployments.
"""
import os
import base64
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# ── Encryption setup ──────────────────────────────────────────────────────────
# SERVER_ENCRYPTION_KEY is a 32-byte key for AES-256-GCM.
# Generated once at server start. Rotating this invalidates all stored sessions.
# In production, load from environment variable so it persists across restarts.

_RAW_KEY = os.getenv("SERVER_ENCRYPTION_KEY", "")
if _RAW_KEY:
    # Decode from base64 if provided as env var
    try:
        _ENC_KEY = base64.b64decode(_RAW_KEY)
        if len(_ENC_KEY) != 32:
            raise ValueError("Key must be 32 bytes")
    except Exception:
        _ENC_KEY = _RAW_KEY.encode()[:32].ljust(32, b'\0')
else:
    # Generate ephemeral key — sessions lost on server restart
    _ENC_KEY = secrets.token_bytes(32)
    logger.warning(
        "SERVER_ENCRYPTION_KEY not set — using ephemeral key. "
        "All sessions will be lost on server restart. "
        "Set SERVER_ENCRYPTION_KEY in .env for persistence."
    )

_AESGCM = AESGCM(_ENC_KEY)

# Session TTL — credentials expire after this duration
SESSION_TTL_HOURS = 24


def _encrypt(plaintext: str) -> str:
    """Encrypt a string with AES-256-GCM. Returns base64-encoded ciphertext."""
    nonce      = secrets.token_bytes(12)    # 96-bit nonce for GCM
    ciphertext = _AESGCM.encrypt(nonce, plaintext.encode(), None)
    # Prepend nonce to ciphertext for storage
    return base64.b64encode(nonce + ciphertext).decode()


def _decrypt(encoded: str) -> str:
    """Decrypt a base64-encoded AES-256-GCM ciphertext."""
    raw        = base64.b64decode(encoded)
    nonce      = raw[:12]
    ciphertext = raw[12:]
    plaintext  = _AESGCM.decrypt(nonce, ciphertext, None)
    return plaintext.decode()


# ── Session store ─────────────────────────────────────────────────────────────
# { session_id: { "api_key_enc": str, "secret_enc": str,
#                 "exchange": str, "testnet": bool,
#                 "balance": float, "connected_at": datetime,
#                 "expires_at": datetime } }

_store: dict[str, dict] = {}


class CredentialStore:

    @staticmethod
    def create_session(
        api_key:    str,
        api_secret: str,
        exchange:   str  = "binance",
        testnet:    bool = False,
        balance:    float = 0.0,
    ) -> str:
        """
        Encrypt and store credentials. Returns a session_id token.
        The session_id is what the frontend stores — never the raw keys.
        """
        session_id = secrets.token_urlsafe(32)
        now        = datetime.now(timezone.utc)

        _store[session_id] = {
            "api_key_enc":  _encrypt(api_key),
            "secret_enc":   _encrypt(api_secret),
            "exchange":     exchange.lower(),
            "testnet":      testnet,
            "balance":      balance,
            "connected_at": now.isoformat(),
            "expires_at":   (now + timedelta(hours=SESSION_TTL_HOURS)).isoformat(),
        }

        logger.info(
            f"Session created: {session_id[:8]}... "
            f"exchange={exchange} testnet={testnet} balance={balance:.2f}"
        )
        return session_id

    @staticmethod
    def get_credentials(session_id: str) -> Optional[dict]:
        """
        Retrieve and decrypt credentials for a session.
        Returns None if session not found or expired.
        """
        record = _store.get(session_id)
        if not record:
            return None

        # Check expiry
        expires_at = datetime.fromisoformat(record["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            CredentialStore.delete_session(session_id)
            logger.info(f"Session {session_id[:8]}... expired and removed.")
            return None

        try:
            return {
                "api_key":    _decrypt(record["api_key_enc"]),
                "api_secret": _decrypt(record["secret_enc"]),
                "exchange":   record["exchange"],
                "testnet":    record["testnet"],
            }
        except Exception as e:
            logger.error(f"Credential decryption failed: {e}")
            return None

    @staticmethod
    def get_session_info(session_id: str) -> Optional[dict]:
        """
        Return non-sensitive session metadata for the frontend.
        Never returns raw or encrypted key material.
        """
        record = _store.get(session_id)
        if not record:
            return None

        expires_at = datetime.fromisoformat(record["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            CredentialStore.delete_session(session_id)
            return None

        return {
            "session_id":   session_id,
            "exchange":     record["exchange"],
            "testnet":      record["testnet"],
            "balance":      record["balance"],
            "connected_at": record["connected_at"],
            "expires_at":   record["expires_at"],
            "connected":    True,
        }

    @staticmethod
    def update_balance(session_id: str, balance: float) -> None:
        """Update cached balance for a session."""
        if session_id in _store:
            _store[session_id]["balance"] = balance

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        existed = session_id in _store
        _store.pop(session_id, None)
        if existed:
            logger.info(f"Session {session_id[:8]}... disconnected.")
        return existed

    @staticmethod
    def is_connected(session_id: str) -> bool:
        """Quick check — is this session valid and not expired?"""
        return CredentialStore.get_session_info(session_id) is not None

    @staticmethod
    def active_session_count() -> int:
        """Number of active sessions (for health endpoint)."""
        return len(_store)