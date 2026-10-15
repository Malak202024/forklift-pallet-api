from pathlib import Path
def coordinates(points, class_id):
    x_values = [x for x, y in points]
    y_values = [y for x, y in points]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return [class_id, center_x, center_y, width, height]

def parse_annotation(line):
    parts=line.split()
    class_id = int(parts[0])
    coordinate_values = [float(value) for value in parts[1:]]
    if len(coordinate_values) == 4:
        return [class_id] + coordinate_values

    points = list(zip(coordinate_values[::2], coordinate_values[1::2]))
    return coordinates(points, class_id)

def convert_label_file(source_path, output_path):
    converted_lines = []

    with open(source_path) as source_file:
        for line in source_file:
            if line.strip():
                box = parse_annotation(line)
                converted_lines.append(" ".join(str(value) for value in box))

    with open(output_path, "w") as output_file:
        output_file.write("\n".join(converted_lines) + "\n")

def convert_label_directory(source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_dir.glob("*.txt"):
        output_path = output_dir / source_path.name
        convert_label_file(source_path, output_path)