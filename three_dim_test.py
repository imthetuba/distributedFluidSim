import numpy as np
from vispy import app, scene
from concurrent.futures import ThreadPoolExecutor
import random
from math import pi

GRAVITY = 900.81
DELTATIME = 0.0002
BOUNDSIZE = 0.7
PARTICLESIZE = 0.05
RADIUSOFINFLUENCE = 1.5
RESTDENSITY = 12
STIFFNESS = 0.8
EXPLOSION_RADIUS = 0.5
EXPLOSION_FORCE = 500.0
MASS = 1.0
DAMPING = 0.95
RESTITUTION = 0.95
MAX_PRESSURE = 10.0
VISCOSITY_COEFFICIENT = 0.1
NUM_PARTICLES = 100
MIN_DISTANCE_PUSH= 100.0


class FluidSimulation3D:
    def __init__(self):
        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True, title='3D Fluid Simulation')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.ArcballCamera(fov=45, distance=5)
        
        # Initialize particle data
        self.n_particles = NUM_PARTICLES
        self.v_position = np.zeros((self.n_particles, 3), dtype=np.float32)
        self.v_velocity = np.zeros((self.n_particles, 3), dtype=np.float32)
        self.spatial_lookup = [None] * self.n_particles
        self.start_indices = {}

        # Explosion state
        self.explosion_active = False
        self.explosion_center = np.array([0.0, -BOUNDSIZE/2 + 0.2, 0.0])  # Bottom of cube
        self.explosion_timer = 0.0
        self.explosion_duration = 0.1  # Duration of explosion effect
        
        
        # Initialize particle positions randomly within bounds
        half_bound = BOUNDSIZE / 2.0
        for i in range(self.n_particles):
            self.v_position[i] = [
                random.uniform(-half_bound, half_bound),
                random.uniform(-half_bound, half_bound),
                random.uniform(-half_bound, half_bound)
            ]
        
        # Create particle sizes and initial colors
        self.particle_sizes = np.full(self.n_particles, PARTICLESIZE , dtype=np.float32)  
        self.particle_colors = np.zeros((self.n_particles, 3), dtype=np.float32)
        self.update_colors()
        
        # Create particle visual using Markers
        self.particle_visual = scene.visuals.Markers(
            pos=self.v_position,
            size=self.particle_sizes,
            face_color=self.particle_colors,
            edge_color='white',
            edge_width=0.5,
            scaling=False,
            spherical=True,
            antialias=1
        )
        self.particle_visual.parent = self.view.scene
        
        # Create boundary wireframe
        self.create_boundary_cube()
        
        # Set up timer for animation
        self.timer = app.Timer(DELTATIME, connect=self.on_timer, start=True)
        
        # Connect keyboard events
        self.canvas.connect(self.on_key_press)

    def create_boundary_cube(self):
        """Create a wireframe cube for 3D boundaries"""
        half_bound = BOUNDSIZE / 2.0
        
        # Define cube vertices
        vertices = np.array([
            [-half_bound, -half_bound, -half_bound],  # 0
            [half_bound, -half_bound, -half_bound],   # 1
            [half_bound, half_bound, -half_bound],    # 2
            [-half_bound, half_bound, -half_bound],   # 3
            [-half_bound, -half_bound, half_bound],   # 4
            [half_bound, -half_bound, half_bound],    # 5
            [half_bound, half_bound, half_bound],     # 6
            [-half_bound, half_bound, half_bound],    # 7
        ], dtype=np.float32)
        
        # Define cube edges as line segments
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Create line visuals for each edge
        for edge in edges:
            line_points = vertices[edge]
            line_visual = scene.visuals.Line(
                pos=line_points,
                color='black',
                width=2
            )
            line_visual.parent = self.view.scene

    def resolve_collisions(self, i):
        half_bound = BOUNDSIZE / 2.0
        
        # Handle collisions for all three dimensions
        for dim in range(3):
            if self.v_position[i, dim] >= half_bound - PARTICLESIZE / 2:
                self.v_position[i, dim] = half_bound - PARTICLESIZE / 2
                self.v_velocity[i, dim] *= -RESTITUTION
            elif self.v_position[i, dim] <= -half_bound + PARTICLESIZE / 2:
                self.v_position[i, dim] = -half_bound + PARTICLESIZE / 2
                self.v_velocity[i, dim] *= -RESTITUTION

    def smoothing_kernel(self, r, h):
        if r > h:
            return 0.0
        factor = 315 / (64 * pi * h**9)
        return factor * (h**2 - r**2)**3

    def smoothing_kernel_gradient(self, r, h):
        if r > h or r == 0:
            return 0.0
        factor = -945 / (32 * pi * h**9)
        return factor * (h**2 - r**2)**2 * r

    def viscosity_kernel(self, r, h):
        if r > h:
            return 0.0
        factor = 45 / (pi * h**6)
        return factor * (h - r)

    def position_to_cell_coord(self, position, radius):
        cellx = int(position[0] / radius)
        celly = int(position[1] / radius)
        cellz = int(position[2] / radius)
        return (cellx, celly, cellz)

    def hash_cell(self, cellx, celly, cellz):
        a = cellx * 15823
        b = celly * 9737333
        c = cellz * 83492791
        return a + b + c

    def update_spatial_lookup(self, points, radius):
        spatial_lookup = []
        for i, position in enumerate(points):
            cellx, celly, cellz = self.position_to_cell_coord(position, radius)
            cell_hash = self.hash_cell(cellx, celly, cellz)
            spatial_lookup.append((cell_hash, i))

        spatial_lookup.sort(key=lambda x: x[0])

        start_indices = {}
        for i, (cell_hash, _) in enumerate(spatial_lookup):
            if cell_hash not in start_indices:
                start_indices[cell_hash] = i

        self.spatial_lookup = spatial_lookup
        self.start_indices = start_indices

    def calculate_density(self, index):
        h = RADIUSOFINFLUENCE
        density = 0
        cellx, celly, cellz = self.position_to_cell_coord(self.v_position[index], PARTICLESIZE)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell_hash = self.hash_cell(cellx + dx, celly + dy, cellz + dz)
                    if neighbor_cell_hash in self.start_indices:
                        start_index = self.start_indices[neighbor_cell_hash]
                        while start_index < len(self.spatial_lookup) and self.spatial_lookup[start_index][0] == neighbor_cell_hash:
                            neighbor_index = self.spatial_lookup[start_index][1]
                            if neighbor_index != index:
                                direction = self.v_position[index] - self.v_position[neighbor_index]
                                distance = np.linalg.norm(direction)
                                if distance <= h:
                                    density += MASS * self.smoothing_kernel(distance, h)
                            start_index += 1
        return density

    def calculate_shared_pressure(self, density_i, density_j):
        pressure_i = STIFFNESS * (density_i - RESTDENSITY)
        pressure_j = STIFFNESS * (density_j - RESTDENSITY)
        shared_pressure = (pressure_i + pressure_j) / 2.0
        return np.clip(shared_pressure, -MAX_PRESSURE, MAX_PRESSURE)

    def find_neighbors(self, i):
        cellx, celly, cellz = self.position_to_cell_coord(self.v_position[i], PARTICLESIZE)
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell_hash = self.hash_cell(cellx + dx, celly + dy, cellz + dz)
                    if neighbor_cell_hash in self.start_indices:
                        start_index = self.start_indices[neighbor_cell_hash]
                        while start_index < len(self.spatial_lookup) and self.spatial_lookup[start_index][0] == neighbor_cell_hash:
                            neighbor_index = self.spatial_lookup[start_index][1]
                            if neighbor_index != i:
                                neighbors.append(neighbor_index)
                            start_index += 1
        return i, neighbors

    def calculate_pressure_force(self, index, neighbors):
        pressureforce = np.zeros(3, dtype=np.float32)
        min_distance = PARTICLESIZE * 0.5
        
        for neighbor_index in neighbors:
            direction = self.v_position[index] - self.v_position[neighbor_index]
            distance = np.linalg.norm(direction)
            if distance <= RADIUSOFINFLUENCE and distance > min_distance:
                slope = self.smoothing_kernel_gradient(distance, PARTICLESIZE)
                density = self.calculate_density(neighbor_index)
                density_of_index = self.calculate_density(index)
                if density > 0:
                    shared_pressure = self.calculate_shared_pressure(density_of_index, density)
                    pressureforce += direction * slope * shared_pressure * MASS / density
            elif distance <= min_distance:
                #add a strong repulsive force to prevent overlap
                pressureforce += direction * MIN_DISTANCE_PUSH  # Arbitrary strong force
        return pressureforce

    def calculate_viscosity_force(self, index, neighbors):
        min_distance = PARTICLESIZE * 0.5
        viscosity_force = np.zeros(3, dtype=np.float32)
        
        for neighbor in neighbors:
            distance = np.linalg.norm(self.v_position[index] - self.v_position[neighbor])
            if distance <= RADIUSOFINFLUENCE and distance > min_distance:
                influence = self.viscosity_kernel(distance, RADIUSOFINFLUENCE)
                viscosity_force += (self.v_velocity[neighbor] - self.v_velocity[index]) * influence
        
        return viscosity_force * VISCOSITY_COEFFICIENT

    def update_colors(self):
        """Update particle colors based on velocity"""
        speed = np.linalg.norm(self.v_velocity, axis=1)
        max_speed = np.max(speed) if np.max(speed) > 0 else 1
        normalized_speed = speed / max_speed
        
        # Create a color gradient from blue (slow) to red (fast)
        self.particle_colors[:, 0] = normalized_speed  # Red channel
        self.particle_colors[:, 1] = 0.3  # Green channel (constant)
        self.particle_colors[:, 2] = 1 - normalized_speed  # Blue channel

    def apply_explosion_force(self):
        """Apply spherical explosion force from the bottom of the cube"""
        for i in range(self.n_particles):
            # Calculate distance from explosion center
            direction = self.v_position[i] - self.explosion_center
            distance = np.linalg.norm(direction)
            
            # Only apply force if particle is within explosion radius
            if distance < EXPLOSION_RADIUS and distance > 0:
                # Normalize direction vector
                direction_normalized = direction / distance
                
                # Calculate force magnitude (inverse square falloff)
                force_magnitude = EXPLOSION_FORCE / (distance**2 + 0.1)  # +0.1 to avoid division by zero
                
                # Apply the explosion force
                explosion_force = direction_normalized * force_magnitude
                self.v_velocity[i] += explosion_force * DELTATIME


    def on_key_press(self, event):
        """Handle keyboard input for camera controls"""
        if event.key == 'r':
            # Reset simulation
            half_bound = BOUNDSIZE / 2.0
            for i in range(self.n_particles):
                self.v_position[i] = [
                    random.uniform(-half_bound, half_bound),
                    random.uniform(-half_bound, half_bound),
                    random.uniform(-half_bound, half_bound)
                ]
            self.v_velocity.fill(0)
        elif event.key == ' ':  # Space bar for explosion
            print("BOOM! Explosion triggered!")
            self.explosion_active = True
            self.explosion_timer = 0.0
        elif event.key == 'p':  # Changed pause to 'p' key
            # Pause/unpause
            if self.timer.running:
                self.timer.stop()
                print("Simulation paused")
            else:
                self.timer.start()
                print("Simulation resumed")

    def on_timer(self, event):
        """Update simulation physics"""
        # Update spatial lookup
        self.update_spatial_lookup(self.v_position, PARTICLESIZE)
        
        # Apply gravity
        self.v_velocity[:, 1] -= GRAVITY * DELTATIME * MASS
        
        if self.explosion_active:
            self.apply_explosion_force()
            self.explosion_timer += DELTATIME
            
            # Deactivate explosion after duration
            if self.explosion_timer >= self.explosion_duration:
                self.explosion_active = False
                print("Explosion finished")
        
        # Calculate forces using parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.find_neighbors, range(self.n_particles)))
        
        # Apply forces
        for i, neighbors in results:
            pressure_force = self.calculate_pressure_force(i, neighbors)
            viscosity_force = self.calculate_viscosity_force(i, neighbors)
            total_force = pressure_force + viscosity_force
            self.v_velocity[i] += total_force * DELTATIME / MASS
        
        # Apply damping
        self.v_velocity *= DAMPING
        
        # Update positions
        self.v_position += self.v_velocity * DELTATIME
        
        # Handle collisions
        with ThreadPoolExecutor() as executor:
            list(executor.map(self.resolve_collisions, range(self.n_particles)))
        
        # Update colors and visual
        self.update_colors()
        self.particle_visual.set_data(
            pos=self.v_position,
            face_color=self.particle_colors
        )

    def run(self):
        """Start the simulation"""
        print("Controls:")
        print("- Mouse: Rotate camera")
        print("- Scroll: Zoom")
        print("- R: Reset simulation")
        print("- Space: Pause/unpause")
        app.run()


if __name__ == '__main__':
    sim = FluidSimulation3D()
    sim.run()