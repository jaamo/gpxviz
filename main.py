import numpy as np
import moderngl
from PIL import Image
from moderngl_window import geometry, WindowConfig
from moderngl_window.conf import settings
import gpxpy
import os
import time as time_module

# Optional: avoid needing config files
settings.WINDOW['class'] = 'moderngl_window.context.pyglet.window.Window'

is_horizontal = False
tracks_folder = 'tracks'

def load_all_gpx_tracks(folder_path):
    all_tracks = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.gpx'):
            filepath = os.path.join(folder_path, filename)
            print(f"Loading {filename}...")
            with open(filepath, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
                coords = []
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            coords.append((point.longitude, point.latitude))
                if coords:
                    all_tracks.append(coords)
    
    return all_tracks

def normalize_coords(coords, min_lon, max_lon, min_lat, max_lat):
    if not coords:
        return []

    def norm(val, min_val, max_val):
        return 2 * (val - min_val) / (max_val - min_val) - 1

    return [(norm(lon, min_lon, max_lon), norm(lat, min_lat, max_lat)) for lon, lat in coords]


class GlowTrackRenderer(WindowConfig):
    gl_version = (3, 3)
    title = "GPX Glow Line"
    # Change this for vertical
    window_size = (1920, 1080) if is_horizontal else (1080, 1920)
    #window_size = (1920, 1080)
    aspect_ratio = None
    resizable = False
    vsync = False
    samples = 4
    resource_dir = "."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_counter = 0
        # Create framebuffer and textures for post-processing
        self.scene_texture = self.ctx.texture(self.window_size, components=4)
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_texture])
 
        self.bright_texture = self.ctx.texture(self.window_size, components=4)
        self.bright_fbo = self.ctx.framebuffer(color_attachments=[self.bright_texture])
 
        self.blur_texture = self.ctx.texture(self.window_size, components=4)
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_texture])
 
        self.quad_fs = geometry.quad_fs()
 
        # Bright-pass shader
        self.bright_prog = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec2 in_position;
            out vec2 uv;
            void main() {
                uv = in_position * 0.5 + 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            uniform sampler2D tex;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                vec3 color = texture(tex, uv).rgb;
                float brightness = dot(color, vec3(0.299, 0.587, 0.114));
                fragColor = brightness > 0.7 ? vec4(color, 1.0) : vec4(0.0);
            }
            '''
        )
 
        # Blur shader (single direction, to be reused)
        self.blur_prog = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec2 in_position;
            out vec2 uv;
            void main() {
                uv = in_position * 0.5 + 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            uniform sampler2D tex;
            uniform vec2 direction;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                float weights[9] = float[](0.3, 0.2, 0.15, 0.1, 0.07,
                                            0.03, 0.01, 0.005, 0.002);
                vec2 tex_offset = direction / vec2(textureSize(tex, 0));
                vec3 result = texture(tex, uv).rgb * weights[0];
                for (int i = 1; i < 9; ++i) {
                    result += texture(tex, uv + tex_offset * i).rgb * weights[i];
                    result += texture(tex, uv - tex_offset * i).rgb * weights[i];
                }
                fragColor = vec4(result, 1.0);
            }
            '''
        )
 
        # Final composite shader
        self.composite_prog = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec2 in_position;
            out vec2 uv;
            void main() {
                uv = in_position * 0.5 + 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            uniform sampler2D scene;
            uniform sampler2D glow;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                vec3 scene_col = texture(scene, uv).rgb;
                vec3 glow_col = texture(glow, uv).rgb;
                fragColor = vec4(scene_col + glow_col, 1.0);
            }
            '''
        )

        # Allow drawing of dots
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE,
        )

        self.vaos = []
        self.age_buffers = []
        self.point_ages_list = []
        self.indices = []


        min_lon, max_lon = 24.301758,25.353355
        min_lat, max_lat = 60.100646,60.380124

        if is_horizontal:
            min_lon, max_lon = 24.301758,25.353355
            min_lat, max_lat = 60.100646,60.380124
        else:
            min_lon, max_lon = 24.581909,25.071487
            min_lat, max_lat = 60.080935,60.487032

        for track in load_all_gpx_tracks(tracks_folder):
            norm_coords = normalize_coords(track, min_lon, max_lon, min_lat, max_lat)
            if not norm_coords:
                continue
            points = np.array(norm_coords, dtype='f4')
            ages = np.zeros(len(points), dtype='f4')

            vbo = self.ctx.buffer(points.tobytes())
            age_buf = self.ctx.buffer(ages.tobytes())
            program = self.get_program()
            vao = self.ctx.vertex_array(
                program,
                [(vbo, '2f', 'in_pos'), (age_buf, '1f', 'in_age')]
            )
            self.program = program

            self.vaos.append(vao)
            self.age_buffers.append(age_buf)
            self.point_ages_list.append(ages)
            self.indices.append(1)  # starting index for this track        

    def get_program(self):
        return self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in float in_age;
            out float age;
            void main() {
                age = in_age;
                gl_PointSize = 5.0;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            in float age;
            out vec4 fragColor;
            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                float glow = smoothstep(0.5, 0.0, dist);
                float alpha = glow * exp(-0.5 * age);  // Fast decay initially
                //alpha = max(alpha, 0.05 * glow);       // Keep faint glow
                fragColor = vec4(0.2, 0.3, 1.0, alpha);
            }
            """
        )

    def on_render(self, time: float, frame_time: float):
        target_frame_time = 1.0 / 30.0
        start_time = time_module.time()
 
        # 1. Render scene to offscreen framebuffer
        self.scene_fbo.use()
        self.scene_fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.line_width = 10.0
        self.frame_counter += 1
        for i, vao in enumerate(self.vaos):
            if self.frame_counter >= i * 5:
                point_ages = self.point_ages_list[i]
                current_index = self.indices[i]
                point_ages[:current_index] += frame_time
                self.age_buffers[i].write(point_ages.tobytes())
                vao.render(mode=moderngl.POINTS, vertices=current_index)
                if current_index < len(point_ages):
                    self.indices[i] = min(self.indices[i] + 150, len(point_ages))
                else:
                    continue  # Skip rendering this track if fully rendered
 
        # 2. Bright-pass filter
        self.bright_fbo.use()
        self.bright_fbo.clear()
        self.scene_texture.use(0)
        self.bright_prog['tex'].value = 0
        self.quad_fs.render(self.bright_prog)
 
        # 3. Blur horizontally
        self.blur_fbo.use()
        self.blur_fbo.clear()
        self.bright_texture.use(0)
        self.blur_prog['tex'].value = 0
        self.blur_prog['direction'].value = (4.0, 0.0)
        self.quad_fs.render(self.blur_prog)
 
        # 4. Blur vertically (back to bright_fbo)
        self.bright_fbo.use()
        self.bright_fbo.clear()
        self.blur_texture.use(0)
        self.blur_prog['tex'].value = 0
        self.blur_prog['direction'].value = (0.0, 4.0)
        self.quad_fs.render(self.blur_prog)
 
        # 5. Final composite
        self.ctx.screen.use()
        self.ctx.screen.clear()
        self.scene_texture.use(0)
        self.bright_texture.use(1)
        self.composite_prog['scene'].value = 0
        self.composite_prog['glow'].value = 1
        self.quad_fs.render(self.composite_prog)

        # Save frame as image
        # image = Image.frombytes('RGBA', self.scene_texture.size, self.scene_texture.read())
        pixels = self.ctx.screen.read(components=3)
        image = Image.frombytes('RGB', self.window_size, pixels)#.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
        frame_name = f"output/{self.frame_counter:04d}.jpg"
        image.save(frame_name, "JPEG")

        elapsed = time_module.time() - start_time
        sleep_time = target_frame_time - elapsed
        if sleep_time > 0:
            time_module.sleep(sleep_time)


if __name__ == '__main__':
    from moderngl_window import run_window_config
    run_window_config(GlowTrackRenderer)