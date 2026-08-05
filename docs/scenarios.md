# Evaluation Scenarios

## Scenario A: Structural Collapse & Fire (Dynamic Evacuation)
**Event:** An explosion on Floor 1 causes a localized fire and structural collapse in the main corridor.
**System Response:** 
- Perception nodes identify fire and debris.
- The `BuildingGraph` updates edge weights to infinity for the collapsed corridor.
- The MAPPO policy diverts evacuees on Floor 1 to secondary exits.
- Smoke propagation triggers pre-emptive evacuation of Floor 2 via stairwells avoiding the affected zone.
**First-Person POV:** *You hear a loud boom. The main hallway fills with smoke. The smart signs instantly change from pointing towards the main lobby to indicating the east stairwell. You follow the crowd safely outside.*

## Scenario B: Mobile Wildlife Threat (Dynamic Avoidance)
**Event:** A dangerous animal enters Floor 2.
**System Response:**
- YOLOv8 detects the animal class and tracks its movement across camera nodes.
- The hazard score of nodes dynamically spikes and decays as the animal moves.
- Router nodes continuously update directional arrows to route evacuees away from the animal's current trajectory.
**First-Person POV:** *An aggressive stray dog runs into the building. The digital signs in your corridor flicker and point you into a secure conference room instead of the hallway where the animal is heading, keeping you out of its path.*

## Scenario C: Active Shooter / Weapon (Sector Lockdown)
**Event:** An individual with a weapon is detected on Floor 3.
**System Response:**
- Threat classification triggers a critical severity alert.
- The system initiates Sector Lockdown protocols.
- Fire doors automatically seal off the affected wing.
- Occupants in the immediate vicinity are directed to hide, while other floors are silently evacuated.
**First-Person POV:** *A lockdown alarm sounds. The corridor doors click locked, and a smart sign inside your office directs you to stay away from the windows and lock the door. Meanwhile, people on the floors below are guided to exit the building quietly through the rear staircases.*
