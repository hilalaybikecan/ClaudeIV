from typing import Optional


def pixel_id_to_well(pixel_id: int) -> str:
    """Convert numeric pixel_id to well notation (e.g., 7 -> A1, 13 -> B1)"""
    # pixel_id = (comp - 1) * 6 + pos
    # pos is 1-6, comp is 1-11
    # So pixel_id ranges from 1 to 66
    if pixel_id < 1 or pixel_id > 66:
        return f"#{pixel_id}"

    # Calculate composition (1-11) and position (1-6)
    comp = (pixel_id - 1) // 6 + 1
    pos = (pixel_id - 1) % 6 + 1

    # Map to well format: rows A-F (position 1-6), columns 1-11 (composition 1-11)
    row_letter = chr(ord("A") + pos - 1)
    col_number = comp

    return f"{row_letter}{col_number}"


def well_to_pixel_id(well: str) -> Optional[int]:
    """Convert well notation to pixel_id (e.g., 'A1' -> 7, 'B1' -> 13)"""
    well = well.strip().upper()
    if len(well) < 2:
        return None

    # Extract row letter and column number
    row_letter = well[0]
    try:
        col_number = int(well[1:])
    except ValueError:
        return None

    # Validate
    if row_letter < "A" or row_letter > "F":
        return None
    if col_number < 1 or col_number > 11:
        return None

    # Convert to position and composition
    pos = ord(row_letter) - ord("A") + 1  # 1-6
    comp = col_number  # 1-11

    # Calculate pixel_id
    pixel_id = (comp - 1) * 6 + pos

    return pixel_id
