import cv2
import numpy as np

def classify_surfaces(image_path, output_path="labeled_output.png", tolerance=0.05):
    """
    Classifies surfaces in a lithic artifact image into Dorsal, Ventral, Platform, and Lateral.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the annotated output image.
        tolerance (float): Percentage tolerance for dimension comparisons (default: 0.05).

    Returns:
        None
    """

    def within_tolerance(value1, value2, tolerance):
        """Checks if two values are approximately equal within a given tolerance."""
        return abs(value1 - value2) / max(value1, value2) <= tolerance

    # Load the image and preprocess
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Threshold the image to binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Extract only the outermost contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (to remove noise)
    min_area = 100  # Minimum contour area to consider
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Extract properties of each contour
    surface_data = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        surface_data.append({
            "index": i,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area,
            "contour": contour
        })

    # Sort surfaces by area (largest to smallest)
    surface_data.sort(key=lambda x: x["area"], reverse=True)

    # Classification rules
    classifications = {}
    if len(surface_data) >= 1:
        # Identify Dorsal (A) and Ventral (B)
        if len(surface_data) > 1:
            height_match = within_tolerance(surface_data[0]["height"], surface_data[1]["height"], tolerance)
            width_match = within_tolerance(surface_data[0]["width"], surface_data[1]["width"], tolerance)
            area_match = within_tolerance(surface_data[0]["area"], surface_data[1]["area"], tolerance)

            if height_match and width_match and area_match:
                classifications[surface_data[0]["index"]] = "Dorsal"
                classifications[surface_data[1]["index"]] = "Ventral"
            else:
                classifications[surface_data[0]["index"]] = "Dorsal"
        else:
            classifications[surface_data[0]["index"]] = "Dorsal"

        # Identify Platform (C)
        for surface in surface_data:
            if surface["index"] not in classifications:
                if (surface["height"] < surface_data[0]["height"] and
                    surface["width"] < surface_data[0]["width"]):
                    classifications[surface["index"]] = "Platform"
                    break

        # Identify Lateral (D)
        for surface in surface_data:
            if surface["index"] not in classifications:
                classifications[surface["index"]] = "Lateral"
                break

    # Draw classifications on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for surface in surface_data:
        label = classifications.get(surface["index"], "Unclassified")
        x, y, w, h = surface["x"], surface["y"], surface["width"], surface["height"]
        color = (0, 255, 0) if label != "Unclassified" else (0, 0, 255)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save and display results
    cv2.imwrite(output_path, output_image)
    print(f"Annotated image saved to {output_path}")
    print("\nSurface Classification Details:")
    for surface in surface_data:
        label = classifications.get(surface["index"], "Unclassified")
        print(f"Surface {surface['index']}: {label}")
        print(f"  - Bounding Box: (x: {surface['x']}, y: {surface['y']}, w: {surface['width']}, h: {surface['height']})")
        print(f"  - Area: {surface['area']}")

# Example usage
if __name__ == "__main__":
    classify_surfaces("data/images/rub_al_khali.png")
