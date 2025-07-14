from PIL import Image
import os
import numpy as np
from collections import deque

already_covered_color = (0, 0, 0)  # Color for covered area
possible_places_color = (195, 195, 195)  # Color for to cover area
to_cover_next_layer_color = (0, 255, 0)  # Color for layer itself

height = None
width = None

include_corners = True
include_corner_vectors = True
heuristic_selection = False
agent_selection = True

if (agent_selection):
    import agent

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = np.array(img)

    global height, width
    height, width, _ = img.shape
    
    already_covered_coords = np.argwhere(np.all(img == already_covered_color, axis=-1))


    possible_places_coords = np.argwhere(np.all(img == possible_places_color, axis=-1))


    to_cover_next_layer_coords = np.argwhere(np.all(img == to_cover_next_layer_color, axis=-1))


    return set([tuple(i.tolist()) for i in already_covered_coords]), set([tuple(i.tolist()) for i in possible_places_coords]), set([tuple(i.tolist()) for i in to_cover_next_layer_coords])
    

already_covered_coords, possible_places_coords, to_cover_next_layer_coords = load_image("test2.png")


print("Already Covered Coordinates:", already_covered_coords)
print("possible_places_coords:", possible_places_coords)
print("Layer Coordinates:", to_cover_next_layer_coords)

def compute_new_layer(to_process_coords):
    for i in range(len(to_process_coords)):
        item=to_process_coords[i]
        to_cover_next_layer_coords.remove(item)
        possible_children = [
            (item[0], item[1] + 1), (item[0], item[1] - 1),
            (item[0] - 1, item[1]), (item[0] + 1, item[1])
        ]
        if include_corners:
            possible_children += [
                (item[0] - 1, item[1] - 1), (item[0] - 1, item[1] + 1),
                (item[0] + 1, item[1] - 1), (item[0] + 1, item[1] + 1)
            ]
        for child in possible_children:
            if child in possible_places_coords and child not in already_covered_coords:
                to_cover_next_layer_coords.add(child)
                already_covered_coords.add(child)
    return list(to_cover_next_layer_coords)

def calculate_vector_layer(layer_coords):
    vectors = []
    for coord in layer_coords:
        possible_parents = [
            (coord[0], coord[1] + 1), (coord[0], coord[1] - 1),
            (coord[0] - 1, coord[1]), (coord[0] + 1, coord[1])
        ]
        possible_parents += [
            (coord[0] - 1, coord[1] - 1), (coord[0] - 1, coord[1] + 1),
            (coord[0] + 1, coord[1] - 1), (coord[0] + 1, coord[1] + 1)
        ]
        curr_vector = (0, 0)
        for parent in possible_parents:
            if parent in already_covered_coords:
                curr_vector = (coord[0] - parent[0], coord[1] - parent[1])
        vectors.append(curr_vector)
    return vectors

def display():
    global height, width
    # Create a new image to visualize the current state
    if height is None or width is None:
        height = max(max(coord[0] for coord in already_covered_coords), 
                    max(coord[0] for coord in possible_places_coords)) + 1
        width = max(max(coord[1] for coord in already_covered_coords), 
                    max(coord[1] for coord in possible_places_coords)) + 1

    # Create white background
    result_img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Set colors for each coordinate set
    for coord in possible_places_coords:
        result_img[coord[0], coord[1]] = possible_places_color

    for coord in already_covered_coords:
        result_img[coord[0], coord[1]] = already_covered_color

    for coord in to_cover_next_layer_coords:
        result_img[coord[0], coord[1]] = to_cover_next_layer_color

    # Convert to PIL Image and save
    result_image = Image.fromarray(result_img)
    result_image.save("result.png")
    print("Image saved as 'result.png'")

import time

def group_nearby_green_spaces():
    """
    Groups green spaces (to_cover_next_layer_coords) that are close to each other.
    Returns a list of lists, where each inner list contains coordinates of a connected group.
    """
    if not to_cover_next_layer_coords:
        return []
    
    # Convert deque to set for faster lookups
    green_coords_set = set(to_cover_next_layer_coords)
    visited = set()
    groups = []
    
    def get_neighbors(coord):
        """Get all adjacent coordinates based on current settings"""
        neighbors = [
            (coord[0], coord[1] + 1), (coord[0], coord[1] - 1),
            (coord[0] - 1, coord[1]), (coord[0] + 1, coord[1])
        ]
        if not include_corners:
            neighbors += [
                (coord[0] - 1, coord[1] - 1), (coord[0] - 1, coord[1] + 1),
                (coord[0] + 1, coord[1] - 1), (coord[0] + 1, coord[1] + 1)
            ]
        return neighbors
    
    def bfs_group(start_coord):
        group = []
        queue = deque([start_coord])
        visited.add(start_coord)
        
        while queue:
            current = queue.popleft()
            group.append(current)
            
            for neighbor in get_neighbors(current):
                if (neighbor in green_coords_set and 
                    neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return group
    
    # Find all connected components
    for coord in green_coords_set:
        if coord not in visited:
            group = bfs_group(coord)
            groups.append(group)
    
    return groups


if __name__ == "__main__":
    time.sleep(3)  # Initial delay to allow for setup
    display()  # Initial display of the image
    responses=[]
    while to_cover_next_layer_coords:
        start_time = time.time()
        green_groups = group_nearby_green_spaces()
        print(f"Green groups: {green_groups}")
        if len(green_groups)==1:
            selected_group = green_groups[0]
        else:
            green_groups = sorted(green_groups, key=lambda g: max(x for x, y in g), reverse=True)
            if not heuristic_selection:
                if (agent_selection):
                    user_prompt = f"Select a group from the following indices: {list(range(len(green_groups)))}"
                    numbered_groups = {i: group for i, group in enumerate(green_groups)}
                    numbered_groups_string = '\n'+'\n'.join(f"{i}: {group}" for i, group in numbered_groups.items())
                    user_prompt += f"\nGreen groups: {numbered_groups_string}."
                    response = agent.get_model_response(user_prompt)
                    responses.append(response)
                    try:
                        selected_group = green_groups[int(response)-1]
                    except (ValueError, IndexError):
                        try:
                            if int(response.strip())==0:
                                selected_group = green_groups[0]
                        except (ValueError, IndexError):
                            print("Invalid input. Defaulting to first group.")
                            selected_group = green_groups[0]
                else:
                    try:
                        selected_group = green_groups[int(input(f"Select group index (0 to {len(green_groups) - 1}): "))]
                    except (ValueError, IndexError):
                        print("Invalid input. Defaulting to first group.")
                        selected_group = green_groups[0]
            else:
                selected_group = green_groups[0]

        layer = compute_new_layer(selected_group)
        vector_layer = calculate_vector_layer(layer)
        print(f"Layer: {layer}")
        print(f"Vectors: {vector_layer}")
        display()
        end_time = time.time()
        print(f"Processed one layer in {end_time - start_time:.2f} seconds")
        if heuristic_selection:
            time.sleep(0.07)
print(responses)
for i in set(responses):
    print(i, responses.count(i))