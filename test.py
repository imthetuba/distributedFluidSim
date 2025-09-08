from vispy import gloo
from vispy import app
import numpy as np
GRAVITY = 9.81
DELTATIME = 0.007
BOUNDSIZE = 22.0
PARTICLESIZE = 10

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
    v_radius = a_size;
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
        n = 50
        self.v_position = np.zeros((n, 2), dtype=np.float32) 
        self.v_velocity = np.zeros((n, 2), dtype=np.float32)
        v_color = np.zeros((n, 3), dtype=np.float32)
        v_size = np.full((n, 1), self.particle_size, dtype=np.float32) 
        for i in range(n):
            self.v_position[i] = [np.random.uniform(-self.bound_size / 2, self.bound_size / 2),
                                  np.random.uniform(-self.bound_size / 2, self.bound_size / 2)]
            v_color[i] = [0, 0, 1]

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        # Set uniform and attribute
        self.program['a_color'] = gloo.VertexBuffer(v_color)
        self.program['a_position'] = gloo.VertexBuffer(self.v_position)
        self.program['a_size'] = gloo.VertexBuffer(v_size)
        gloo.set_state(clear_color='white', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        # Start a timer to update the canvas
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)



    def resolve_collisions(self):
        half_bound = (self.bound_size / 2.0 )- self.particle_size
        for i in range(1, len(self.v_position)):
            if self.v_position[i, 1] > half_bound: 
                self.v_position[i, 1] = half_bound
                self.v_velocity[i, 1] *= -1  
            elif self.v_position[i, 1] < -half_bound:  
                self.v_position[i, 1] = -half_bound
                self.v_velocity[i, 1] *= -1  

            if self.v_position[i, 0] > half_bound:
                self.v_position[i, 0] = half_bound
                self.v_velocity[i, 0] *= -1
            elif self.v_position[i, 0] < -half_bound:
                self.v_position[i, 0] = -half_bound
                self.v_velocity[i, 0] *= -1
           
    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program['a_position'].set_data(self.v_position)  # Update position
        self.program.draw('points')


    def on_timer(self, event):
        self.v_velocity[:, 1] -= GRAVITY * DELTATIME
        self.v_position += self.v_velocity * DELTATIME 
        self.resolve_collisions()
        self.update()


if __name__ == '__main__':
    canvas = Canvas()
    app.run()