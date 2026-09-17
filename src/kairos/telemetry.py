"""Logging and telemetry with a guard against note text reaching logs.

Rule 0.3: note text is never written to logs. The guard truncates any log message longer
than ``MAX_MESSAGE_CHARS`` and drops records that carry a ``note_text`` attribute, so even
an accidental ``logger.info(text)`` cannot leak a full note. Azure Monitor export is enabled
only when a connection string is configured; sampling is on.
"""
from __future__ import annotations

import logging
import os

MAX_MESSAGE_CHARS = 400


class NoteTextGuard(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "note_text"):
            return False
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        if len(msg) > MAX_MESSAGE_CHARS:
            record.msg = msg[:MAX_MESSAGE_CHARS] + " ...[truncated by NoteTextGuard]"
            record.args = ()
        return True


_configured = False


def configure_telemetry(service_name: str) -> None:
    """Idempotent: configure stdlib logging, add the guard, attach Azure Monitor if set."""
    global _configured
    if _configured:
        return
    logging.basicConfig(level=os.environ.get("KAIROS_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    guard = NoteTextGuard()
    for h in logging.getLogger().handlers:
        h.addFilter(guard)
    logging.getLogger("uvicorn.access").addFilter(guard)
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if conn:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(connection_string=conn, logger_name="kairos",
                                    enable_live_metrics=False,
                                    resource_attributes={"service.name": service_name})
            logging.getLogger("kairos").addFilter(guard)
        except Exception as ex:  # noqa: BLE001 - telemetry must never break a service
            logging.getLogger(__name__).warning("Azure Monitor not configured: %s", type(ex).__name__)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.addFilter(NoteTextGuard())
    return lg
