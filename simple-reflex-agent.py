import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend to avoid Tkinter/Tcl issues on Windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

plt.ion()  # Enable interactive mode so windows open/close automatically without blocking

# Tracks the cleanliness state of each room (Clean or Dirty)
environment = {
    "Room1": "Clean",
    "Room2": "Dirty",
    "Room3": "Clean",
    "Room4": "Clean"
}

# Grid coordinates (col, row) for each room in the 2x2 visualization
room_positions = {
    "Room1": (0, 0),
    "Room2": (1, 0),
    "Room3": (0, 1),
    "Room4": (1, 1)
}

rooms = list(environment.keys())  # Ordered list of room names for indexed access
agent_index = 0  # Starting position of the agent (index into rooms list)


def agent(state):
    """Simple reflex agent: decides action based solely on the current room's state."""
    if state == "Dirty":
        return "Clean"  # Clean the room if it is dirty
    else:
        return "Move"   # Move to the next room if already clean


def build_environment(env, agent_pos, step):
    """Draws the 2x2 room grid, colors rooms by state, and marks the agent's position."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)   # Grid spans 2 units wide
    ax.set_ylim(0, 2)   # Grid spans 2 units tall
    ax.set_xticks([])   # Hide x-axis tick marks
    ax.set_yticks([])   # Hide y-axis tick marks
    ax.set_title(f"Step {step} - Agent in {rooms[agent_pos]}")

    # Draw each room as a colored rectangle (green = Clean, red = Dirty)
    for room, pos in room_positions.items():
        x, y = pos
        color = 'green' if env[room] == "Clean" else 'red'
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        plt.text(x + 0.5, y + 0.5, room, ha='center', va='center', fontsize=12)  # Room label

    # Draw the agent as a blue circle centered in the current room
    agent_x, agent_y = room_positions[rooms[agent_pos]]
    agent_circle = patches.Circle((agent_x + 0.5, agent_y + 0.5), 0.2, color='blue', alpha=0.5)
    ax.add_patch(agent_circle)

    plt.pause(1)   # Display the window for 1 second
    plt.close()    # Close the window before moving to the next step


def run_agent():
    """Runs the agent through all rooms sequentially, cleaning dirty ones."""
    step = 0
    for i in range(len(rooms)):
        current_room = rooms[i]                        # Room the agent is currently in
        action = agent(environment[current_room])      # Decide action based on room state
        print(f"Step {step} | Room: {current_room} | State: {environment[current_room]} | Action: {action}")
        build_environment(environment, i, step)        # Visualize current state
        if action == "Clean":
            environment[current_room] = "Clean"        # Update room state after cleaning
        step += 1
    print("All rooms visited.")


run_agent()
