import json
import redis
import threading
from typing import Callable, Optional
from core.messaging.schemas import (
    EnvironmentStateMsg, HazardUpdateMsg, PolicyActionMsg,
    DisasterInjectionMsg, SimControlMsg
)
from core.messaging.constants import (
    CHANNEL_ENV_STATE, CHANNEL_HAZARD_UPDATE,
    CHANNEL_POLICY_ACTION, CHANNEL_GUI_CONTROL,
    CHANNEL_DISASTER_INJECT
)

class EnvironmentEventBus:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.pubsub = self.redis_client.pubsub()
        self.listener_thread: Optional[threading.Thread] = None
        self._running = False
        self.callbacks = {}

    def publish_state(self, state: EnvironmentStateMsg):
        self.redis_client.publish(CHANNEL_ENV_STATE, state.model_dump_json())

    def subscribe_hazard_updates(self, callback: Callable[[HazardUpdateMsg], None]):
        self._register_callback(CHANNEL_HAZARD_UPDATE, callback, HazardUpdateMsg)

    def subscribe_policy_actions(self, callback: Callable[[PolicyActionMsg], None]):
        self._register_callback(CHANNEL_POLICY_ACTION, callback, PolicyActionMsg)

    def subscribe_control(self, callback: Callable[[SimControlMsg], None]):
        self._register_callback(CHANNEL_GUI_CONTROL, callback, SimControlMsg)

    def subscribe_disaster_injection(self, callback: Callable[[DisasterInjectionMsg], None]):
        self._register_callback(CHANNEL_DISASTER_INJECT, callback, DisasterInjectionMsg)

    def _register_callback(self, channel: str, callback: Callable, schema_cls):
        if channel not in self.callbacks:
            self.callbacks[channel] = []
            self.pubsub.subscribe(channel)
        self.callbacks[channel].append((callback, schema_cls))

    def start_listening(self):
        if self._running:
            return
        self._running = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        for message in self.pubsub.listen():
            if not self._running:
                break
            if message['type'] == 'message':
                channel = message['channel'].decode('utf-8')
                data_str = message['data'].decode('utf-8')
                
                if channel in self.callbacks:
                    for callback, schema_cls in self.callbacks[channel]:
                        try:
                            # Parse with pydantic
                            parsed = schema_cls.model_validate_json(data_str)
                            callback(parsed)
                        except Exception as e:
                            print(f"Error parsing or executing callback on {channel}: {e}")

    def stop_listening(self):
        self._running = False
        if self.listener_thread:
            # We can issue an unsubscribe to break the listen loop faster
            self.pubsub.unsubscribe()
            self.listener_thread.join(timeout=1.0)
            
    def close(self):
        self.stop_listening()
        self.pubsub.close()
        self.redis_client.close()

if __name__ == "__main__":
    # Test script entrypoint
    pass
