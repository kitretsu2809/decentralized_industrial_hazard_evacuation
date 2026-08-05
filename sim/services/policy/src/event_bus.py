import json
import logging
import threading
from typing import Callable, Optional
import redis
from pydantic import ValidationError

from core.messaging.schemas import EnvironmentStateMsg, PolicyActionMsg
from core.messaging.constants import CHANNEL_ENV_STATE, CHANNEL_POLICY_ACTION

logger = logging.getLogger(__name__)

class PolicyEventBus:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._env_state_callback: Optional[Callable[[EnvironmentStateMsg], None]] = None

    def subscribe_env_state(self, callback: Callable[[EnvironmentStateMsg], None]):
        self._env_state_callback = callback

    def publish_action(self, action: PolicyActionMsg):
        try:
            payload = action.model_dump_json()
            self.redis_client.publish(CHANNEL_POLICY_ACTION, payload)
        except Exception as e:
            logger.error(f"Failed to publish policy action: {e}")

    def start_listening(self):
        if self._running:
            return
            
        self.pubsub.subscribe(**{CHANNEL_ENV_STATE: self._handle_env_state})
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        logger.info("Policy event bus listener started.")

    def stop_listening(self):
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
        self.pubsub.unsubscribe(CHANNEL_ENV_STATE)
        logger.info("Policy event bus listener stopped.")

    def close(self):
        self.stop_listening()
        self.pubsub.close()
        self.redis_client.close()

    def _listen_loop(self):
        while self._running:
            try:
                self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except Exception as e:
                logger.error(f"Error in pubsub listen loop: {e}")

    def _handle_env_state(self, message: dict):
        if not self._env_state_callback:
            return
            
        try:
            data = message.get('data')
            if data:
                state_msg = EnvironmentStateMsg.model_validate_json(data)
                self._env_state_callback(state_msg)
        except ValidationError as e:
            logger.error(f"Validation error parsing EnvironmentStateMsg: {e}")
        except Exception as e:
            logger.error(f"Error handling environment state message: {e}")
