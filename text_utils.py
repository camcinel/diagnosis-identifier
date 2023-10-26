from typing import Tuple


def contains_category(text: str, category: Tuple[str, ...]) -> str | None:
    category += tuple([x.upper() for x in category])

    lines = text.split('\n')
    for index, line in enumerate(lines):
        for cat in category:
            if cat in line:
                return lines[index + 1].strip()
    return None


def get_category(text: str, category: Tuple[str, ...]) -> str | None:
    category += tuple([x.upper() for x in category])

    categories = text.split('\n\n')
    category_chunks = [chunk.strip() for chunk in categories if chunk.strip().startswith(category)]
    if category_chunks:
        category_chunk = category_chunks[0]
        for cat in category:
            category_chunk = category_chunk.replace(cat, '')
        category_chunk = category_chunk.strip(':').strip()
    else:
        category_chunk = None
    return category_chunk
