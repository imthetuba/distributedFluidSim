from vispy import gloo
from vispy import app
from vispy.scene import SceneCanvas
import numpy as np
import random
GRAVITY = 9.81
DELTATIME = 0.0016
BOUNDSIZE = 1.5
PARTICLESIZE = 0.1
RADIUSOFINFLUENCE = 1.5
RESTDENSITY = 8
STIFFNESS = 1
MASS = 1.0
DAMPING = 0.8
RESTITUTION = 0.5

VERT_SHADER = """
attribute vec2  a_position;
attribute vec3  a_color;
attribute float a_size;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_radius = a_size * 100.0;
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(0.0,0.0,0.0,0.5);
    v_bg_color  = vec4(a_color,    1.0);

    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
}
"""

FRAG_SHADER = """
#version 120

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;
void main()
{
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > v_radius)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')
        self.ps = self.pixel_scale
        self.particle_size = PARTICLESIZE * self.ps
        self.bound_size = BOUNDSIZE * self.ps

        # Create vertices
        n = 30
        self.v_position = np.zeros((n, 2), dtype=np.float32) 
        self.v_velocity = np.zeros((n, 2), dtype=np.float32)
        v_color = np.zeros((n, 3), dtype=np.float32)
        v_size = np.full((n, 1), self.particle_size, dtype=np.float32)
        grid_size = int(np.sqrt(n))  
        spacing = self.bound_size / (grid_size + 2)
        index = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if index <= n: 
                    self.v_position[index] = [
                        -self.bound_size / 2 + i * spacing + spacing / 2,
                        -self.bound_size / 2 + j * spacing + spacing / 2,
                    ]
                    index += 1
        print("Particle positions:", self.v_position)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        # Set uniform and attribute
        self.program['a_color'] = gloo.VertexBuffer(v_color)
        self.program['a_position'] = gloo.VertexBuffer(self.v_position)
        self.program['a_size'] = gloo.VertexBuffer(v_size)

        half_bound = self.bound_size / 2.0
        boundary_vertices = np.array([
            [-half_bound, -half_bound],
            [half_bound, -half_bound],
            [half_bound, half_bound],
            [-half_bound, half_bound],
            [-half_bound, -half_bound]  # Close the loop
        ], dtype=np.float32)

        self.boundary_program = gloo.Program(
            "attribute vec2 a_position; void main() { gl_Position = vec4(a_position, 0.0, 1.0); }",
            "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"
        )
        self.boundary_program['a_position'] = gloo.VertexBuffer(boundary_vertices)


        gloo.set_state(clear_color='white', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        # Start a timer to update the canvas
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)



    def resolve_collisions(self):
        half_bound = (self.bound_size / 2.0 )
        for i in range(1, len(self.v_position)):
            if self.v_position[i, 1] >= half_bound: 
                # if above upper boundary
                self.v_position[i, 1] = half_bound - self.particle_size / 2
                self.v_velocity[i, 1] *= -RESTITUTION  
            elif self.v_position[i, 1] <= -half_bound:  
                # if below lower boundary
                self.v_position[i, 1] = -half_bound+ self.particle_size/ 2
                self.v_velocity[i, 1] *= -RESTITUTION 

            if self.v_position[i, 0] >= half_bound:
                # if beyond right boundary
                self.v_position[i, 0] = half_bound - self.particle_size / 2
                self.v_velocity[i, 0] *= -RESTITUTION
            elif self.v_position[i, 0] <= -half_bound:
                # if beyond left boundary
                self.v_position[i, 0] = -half_bound+self.particle_size / 2
                self.v_velocity[i, 0] *= -RESTITUTION
           

    def smoothing_kernel(self, r, h):
        if r >= 0 and r <= h:
            return  15 / (np.pi * h**6 ) * (h - r)**3
        else:
            return 0
    def smoothing_kernel_gradient(self, r, h):
        if r > 0 and r <= h:
            return  -45 / (np.pi * h**6) * (h - r)**2  / r
        else:
            return np.array([0.0, 0.0])

    def calculate_density(self, index):
        h = RADIUSOFINFLUENCE * self.ps  
        density = 0
        for particle in range(len(self.v_position)):
            if particle != index:  
                direction = self.v_position[index] - self.v_position[particle]
                distance = np.linalg.norm(direction)
                if distance <= h:  
                    # Use the smoothing kernel for density calculation
                    density += MASS * self.smoothing_kernel(distance, h)
        return density
    
    def calculate_shared_pressure(self, density_i, density_j):
        pressure_i = STIFFNESS * (density_i - RESTDENSITY)
        pressure_j = STIFFNESS * (density_j - RESTDENSITY)
        shared_pressure = (pressure_i + pressure_j) / 2.0
        return shared_pressure

    def calculate_pressure_force(self, index):  
        pressureforce = np.zeros(2, dtype=np.float32)
        for particle in range(len(self.v_position)):
            if particle != index:
                direction = self.v_position[index] - self.v_position[particle]
                distance = np.linalg.norm(direction)
                if distance <= RADIUSOFINFLUENCE * self.ps:
                    slope = self.smoothing_kernel_gradient(distance, self.particle_size)
                    density = self.calculate_density(particle)
                    density_of_index = self.calculate_density(index)
                    if density > 0:
                        shared_pressure = self.calculate_shared_pressure(density_of_index, density)
                        random_perturbation = np.array([random.uniform(-0.001, 0.001), random.uniform(-0.001, 0.001)])
                        pressureforce += random_perturbation + direction * slope * shared_pressure * MASS / density
                    
        return pressureforce
    

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program['a_position'].set_data(self.v_position)  # Update position
        self.program.draw('points')
        self.boundary_program.draw('line_strip')


    def on_timer(self, event):
        self.v_velocity[:, 1] -= GRAVITY * DELTATIME *MASS

        for i in range(len(self.v_position)):
            pressure_force = self.calculate_pressure_force(i)
            self.v_velocity[i] += pressure_force * DELTATIME / MASS  

        self.v_velocity *= DAMPING

        self.v_position += self.v_velocity * DELTATIME


        self.resolve_collisions()

        self.update()


if __name__ == '__main__':
    canvas = Canvas()
    app.run()