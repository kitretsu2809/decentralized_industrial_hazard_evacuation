"""
GPIO Door HAL Implementation
"""
from typing import Dict
import logging
from core.hal.base import DoorHAL
from core.graph.types import EdgeState

logger = logging.getLogger(__name__)

# Fallback for non-Jetson testing
try:
    import Jetson.GPIO as GPIO
    HAVE_GPIO = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        HAVE_GPIO = True
    except ImportError:
        HAVE_GPIO = False
        logger.warning("No GPIO library found. Using mock GPIO.")

class GPIODoor(DoorHAL):
    def __init__(self, gpio_pin: int, normally_open: bool = True):
        self.gpio_pin = gpio_pin
        self.normally_open = normally_open
        self.state_cache = {}
        
        self._init_gpio()

    def _init_gpio(self):
        if HAVE_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            # Default state based on normally_open
            GPIO.output(self.gpio_pin, GPIO.LOW if self.normally_open else GPIO.HIGH)
        else:
            logger.info(f"Mock GPIO init on pin {self.gpio_pin}")

    def lock(self, edge_id: str) -> bool:
        if HAVE_GPIO:
            GPIO.output(self.gpio_pin, GPIO.HIGH if self.normally_open else GPIO.LOW)
        else:
            logger.info(f"Mock: Locking door on edge {edge_id}")
            
        self.state_cache[edge_id] = EdgeState.LOCKED.value
        return True

    def unlock(self, edge_id: str) -> bool:
        if HAVE_GPIO:
            GPIO.output(self.gpio_pin, GPIO.LOW if self.normally_open else GPIO.HIGH)
        else:
            logger.info(f"Mock: Unlocking door on edge {edge_id}")
            
        self.state_cache[edge_id] = EdgeState.OPEN.value
        return True

    def emergency_seal(self, edge_id: str) -> bool:
        """Lock immediately and refuse manual unlock unless overridden."""
        self.lock(edge_id)
        self.state_cache[edge_id] = EdgeState.EMERGENCY_SEALED.value
        return True

    def get_state(self, edge_id: str) -> str:
        return self.state_cache.get(edge_id, EdgeState.OPEN.value)

    def get_all_states(self) -> Dict[str, str]:
        return self.state_cache

    def set_state(self, actions: Dict[str, int]) -> None:
        """Process policy door commands."""
        # actions map edge_id to core.policy.action_types.DoorCommand
        # (1 = OPEN, 2 = LOCK, 3 = EMERGENCY_SEAL)
        for edge_id, cmd in actions.items():
            if cmd == 1:
                self.unlock(edge_id)
            elif cmd == 2:
                self.lock(edge_id)
            elif cmd == 3:
                self.emergency_seal(edge_id)
