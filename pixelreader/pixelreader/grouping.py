def comp_to_group(comp: int) -> int:
    if comp in (1, 2):
        return 1
    if comp in (10, 11):
        return 9
    return comp - 1  # 3..9 -> 2..8
