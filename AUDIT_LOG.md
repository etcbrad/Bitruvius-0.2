
# Bitruvius Design Audit Log

## [v1.0] - Lockdown Milestone
- **Head**: Implemented "Flattened Head" design. Inverted wedge shape with base width (neck) at 25% of unit and top width (skull) at 55%.
- **Shoulders**: Adjusted RIGGING to lower shoulder anchors by one `ANATOMY.COLLAR` length.
- **Hands/Feet**: Redesigned `hand-foot-arrowhead-shape` with a ultra-thin base (30% width).
- **Parenting**: Confirmed Olive Collar is the parent for Head, RShoulder, and LShoulder.

## [v1.5] - Dynamic Joint Constraints
- **Joint Modes**: Added 5 behavioral modes for joints, cycled via Double-Click:
    - **Locked (Default)**: Standard FK.
    - **Stretch**: Child counter-rotates (-1:1).
    - **Curl**: Child rotates with parent (+1:1).
    - **Tense**: Slight resistance (-0.5:1).
    - **Loose**: Lazy follow (+0.5:1).

## [v1.6] - Chain Selection
- **Triple-Click Interaction**: Detects triple-clicking segments to select logical groups (Arms, Legs, Core).

## [v1.7] - Unified Command HUD
- **Command HUD**: Implemented persistent legend and status monitor.

## [v1.8] - Simplified Interaction Model
- **Lock Removal**: Completely removed the "Freeze/Lock Joint" feature and UI icon to reduce cognitive load and streamline posing.
- **Interaction Hierarchy**:
    - **Single-Click**: Select part for individual adjustment.
    - **Double-Click**: Cycle constraint behavior (HUD updates to show mode).
    - **Triple-Click**: Select entire logical chain (e.g., full arm).
- **HUD Update**: Updated legend to reflect the removal of locking and the focus on kinetic behaviors.