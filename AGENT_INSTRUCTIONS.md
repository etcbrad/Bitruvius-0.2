

# AGENT IMPLEMENTATION SPECIFICATION: MASK & BACKGROUND

**OBJECTIVE:** Implement robust file upload and transform controls for a mask overlay and a background image.

**ZERO-TOLERANCE POLICY:** Deviations from this technical specification are forbidden. Do not create new state variables or modify component structure beyond the explicit instructions provided. Adhere strictly to the existing element IDs and state variable names.

---

### **TASK 1: ACTIVATE MASK OVERLAY CONTROLS**

1.  **TARGET FILE:** `App.tsx`
2.  **LOCATE ELEMENT:** Find the `div` with the `id="mask-controls-placeholder"`.
3.  **ACTION:** Remove the `hidden` CSS class from this `div` to make the controls visible.
4.  **STATE WIRING:**
    *   **File Input:** The `<input type="file" id="mask-upload">` already triggers the `handleMaskUpload` function via its `<label>`. This function correctly uses `FileReader.readAsDataURL` to update the `maskImage` state. **Verify this behavior is preserved.**
    *   **Transform Sliders:**
        *   Connect the "X Offset" slider to update the `x` property of the `maskTransform` state.
        *   Connect the "Y Offset" slider to update the `y` property of the `maskTransform` state.
        *   Connect the "Rotation" slider to update the `rotation` property of the `maskTransform` state.
        *   Connect the "Scale" slider to update the `scale` property of the `maskTransform` state.
    *   **Remove Button:** The "Remove Mask" button should set the `maskImage` state to `null`.
5.  **RENDERING VERIFICATION (NO CODE CHANGE NEEDED):**
    *   Confirm that the `Mannequin` component receives the `maskImage` and `maskTransform` props.
    *   Confirm that within `Mannequin.tsx`, an `<image>` element is rendered inside the `head` group, using `maskTransform` to position, scale, and rotate it.

---

### **TASK 2: IMPLEMENT BACKGROUND IMAGE SYSTEM**

1.  **TARGET FILE:** `App.tsx`
2.  **STATE VARIABLES (ALREADY DECLARED):**
    *   `backgroundImage: string | null`
    *   `backgroundTransform: { x: number; y: number; rotation: number; scale: number; }`
    *   **Do not create new state variables.** Use these.
3.  **LOCATE UI ELEMENT:** Find the `div` with the `id="background-controls-placeholder"`.
4.  **ACTION:** Remove the `hidden` CSS class from this `div`.
5.  **IMPLEMENT HANDLER FUNCTION:**
    *   Locate the placeholder function `handleBackgroundUpload`.
    *   Implement it using the exact same logic as `handleMaskUpload`, but it must update the `backgroundImage` state variable.
    *   **CRITICAL:** The function must handle the `ChangeEvent` from a file input and use `FileReader.readAsDataURL`.
6.  **STATE WIRING:**
    *   **File Input:** The `<label htmlFor="background-upload">` should trigger the file input, which in turn must call your implemented `handleBackgroundUpload` function.
    *   **Transform Sliders:** Wire up the four sliders inside the placeholder `div` to control the `x`, `y`, `rotation`, and `scale` properties of the `backgroundTransform` state object, respectively.
    *   **Remove Button:** The "Remove Background" button should set the `backgroundImage` state to `null`.
7.  **RENDERING LOGIC:**
    *   Locate the main `<svg>` element in `App.tsx`.
    *   Inside the `<svg>`, find the `<image id="background-image-renderer">`. It is currently commented out or hidden.
    *   **ACTION:** Ensure this `<image>` element is rendered if `backgroundImage` is not null.
    *   **ATTRIBUTES:**
        *   `href` must be bound to the `backgroundImage` state.
        *   `transform` must be dynamically constructed from the `backgroundTransform` state: `translate(x, y) rotate(rotation) scale(scale)`.
        *   This element **MUST** be rendered *before* the main mannequin `<g>` group to ensure it is in the background.
        *   Set placeholder `width` and `height` (e.g., "1000") and position it at `x="-500"` `y="-500"` to center it before transforms are applied.