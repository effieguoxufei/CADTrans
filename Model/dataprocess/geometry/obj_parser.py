import os
import sys
import numpy as np

from geometry.arc import Arc
from geometry.circle import Circle
from geometry.line import Line

from geometry import geom_utils
import pdb
import os
import math
import random
import io
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

class ColorGenerator:
    @staticmethod
    def get_random_color():
        r = lambda: random.randint(0, 255)
        return "#%02X%02X%02X" % (r(), r(), r())

class Geometry:
    def __init__(self, shape=''):
        self.shape = shape

    @property
    def draw(self, image: Image, start_coords: Tuple[int, int], end_coords: Tuple[int, int], radius: float, color: str):
        raise NotImplementedError

class Line(Geometry):
    @property
    def draw(self, image: Image, start_coords: Tuple[int, int], end_coords: Tuple[int, int], radius: float, color: str):
        draw = ImageDraw.Draw(image)
        draw.line([start_coords, end_coords], width=2, fill=color)

class Circle(Geometry):
    @property
    def draw(self, image: Image, center_coords: Tuple[int, int], radius: float, color: str):
        img_width, img_height = image.size
        mask = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((center_coords[0]-radius, center_coords[1]-radius, center_coords[0]+radius, center_coords[1]+radius), outline=255, width=2)
        image.putalpha(mask)
        image.paste(Image.new('RGBA', size=(radius*2, radius*2), color=color), (center_coords[0]-radius, center_coords[1]-radius))

class Arc(Geometry):
    @property
    def draw(self, image: Image, start_coords: Tuple[int, int], mid_coords: Tuple[int, int], end_coords: Tuple[int, int], radius: float, color: str):
        angle = math.atan2(end_coords[1]-mid_coords[1], end_coords[0]-mid_coords[0]) \
              - math.atan2(start_coords[1]-mid_coords[1], start_coords[0]-mid_coords[0])
        angle = abs(angle)

        img_width, img_height = image.size
        mask = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(mask)
        draw.pieslice(((min(mid_coords[0], min(start_coords[0], end_coords[0]))-radius)-10, (min(mid_coords[1], min(start_coords[1], end_coords[1]))-radius)-10, (max(mid_coords[0], max(start_coords[0], end_coords[0]))+radius)+10, (max(mid_coords[1], max(start_coords[1], end_coords[1]))+radius)+10), start=math.degrees(-angle)*-1, end=math.degrees(angle)*-1, fill=255)
        image.putalpha(mask)
        image.paste(Image.new('RGBA', size=(radius*2, radius*2), color=color), ((min(mid_coords[0], min(start_coords[0], end_coords[0]))-radius)-10, (min(mid_coords[1], min(start_coords[1], end_coords[1]))-radius)-10))


class OBJParser:
    """
    A class to read an OBJ file containing the sketch data
    and hand it back in a form which is easy to work with.
    """
    def __init__(self, pathname=None):
        self.pathname = pathname


    def convert_vertices(self, vertices):
        """Convert all the vertices to .obj format"""
        vertex_strings = ""
        for pt in vertices:
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


    def convert_curves(self, faces):
        curve_strings = ""
        total_curve = 0

        # Faces (multiple closed regions)
        for group_idx, loops in enumerate(faces):
            curve_strings += f"\nface\n"
            # Multiple loops (inner and outer)
            for loop in loops: 
                if loop[0].is_outer:  
                    curve_strings += f"out\n"
                else:
                    curve_strings += f"in\n"
                # All curves in one loop
                for curve in loop:
                    total_curve += 1
                    if curve.type == 'line':
                        curve_strings += f"l {curve.start_idx} {curve.end_idx}\n"
                    elif curve.type == 'circle':
                        curve_strings += f"c {curve.center_idx} {curve.radius_idx}\n"
                    elif curve.type == 'arc':
                        curve_strings += f"a {curve.start_idx} {curve.mid_idx} {curve.center_idx} {curve.end_idx}\n"

        return curve_strings, total_curve


    def parse3d(self, point3d):
        x = point3d[0]
        y = point3d[1]
        z = point3d[2]
        return str(x)+' '+str(y)+' '+str(z)



    def parse_sketch(self, scale=1.0, save_images=False, export_folder='./output') -> List[Tuple[Optional[str], Image.Image]]:
        """Parse obj file and generate geometries.
           Save them as png files when save_images is True."""

        assert self.pathname is not None, "File is None"
        assert self.pathname.exists(), "No such file"

        # Initialize lists to store parsed geometries
        geoms = []

        # Read vertices
        vertex_list = []
        with open(self.pathname) as obj_file:
            for line in obj_file:
                tokens = line.split()
                if not tokens:
                    continue
                line_type = tokens[0]
                if line_type == "v":
                    vertex_list.append([float(x) for x in tokens[1:]])

        vertices = np.array(vertex_list, dtype=np.float64) * scale

        # Reset file pointer
        obj_file.seek(0)

        # Read curves
        lines = []
        shapes = {}
        idx = 0
        while idx < len(vertices):
            line = obj_file.readline().decode().strip()
            if not line:
                break

            if line.startswith("f"):
                parts = line.split()[1:]
                part_count = len(parts)
                first_part = tuple(map(lambda x: map(int, x.split("/")), parts[0].split()))

                shape = next((k for k, v in shapes.items() if v == first_part[-1]), None)
                if shape is None:
                    shape = f'shape_{len(shapes)}'
                    shapes[shape] = first_part[-1]

                geometry = Geometry(shape)
                geoms.append((shape, geometry))
                idx += part_count
                continue

            coords = list(map(lambda x: tuple(map(lambda y: int(round(y)), filter(None, x.split('/')))), line.split()[1:]))
            prev_coords = coords[0]
            points = []

            for coord in coords[1:]:
                x, y = vertices[coord[0]-1][::-1]
                points.append((x, y))

                if coord[1] == '-1':
                    continue

                if save_images:
                    img = Image.new('RGB', (2*max(abs(prev_coords[0]), abs(x))+1, 2*max(abs(prev_coords[1]), abs(y))+1), 'white')
                    drawer = eval(geoms[-1][1].__class__.__name__ + ".draw")
                    drawer(img, prev_coords, coord, 1, ColorGenerator.get_random_color())
                    filename = os.path.join(export_folder, f"{geoms[-1][0]}.png")
                    img.save(filename)

                prev_coords = coord

            if save_images:
                img = Image.new('RGB', (2*max(abs(prev_coords[0]), abs(x))+1, 2*max(abs(prev_coords[1]), abs(y))+1), 'white')
                drawer = eval(geoms[-1][1].__class__.__name__ + ".draw")
                drawer(img, prev_coords, coords[-1], 1, ColorGenerator.get_random_color())
                filename = os.path.join(export_folder, f"{geoms[-1][0]}_final.png")
                img.save(filename)

        return geoms
    


    def write_obj2(self, file, vertices, faces, meta_info, scale=None):
        """ Write to .obj file """
        vertex_strings = self.convert_vertices(vertices)
        curve_strings, total_curve = self.convert_curves(faces)
        
        with open(file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write(f"# Vertices: {len(vertices)}\n")
            fh.write(f"# Curves: {total_curve}\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            fh.write("ExtrudeOperation: " + meta_info['set_op']+"\n")
            extrude_string = 'Extrude '
            for value in meta_info['extrude_value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")
        
            #Write refe plane transformation 
            p_orig = self.parse3d(meta_info['t_orig'])
            x_axis = self.parse3d(meta_info['t_x'])
            y_axis = self.parse3d(meta_info['t_y'])
            z_axis = self.parse3d(meta_info['t_z'])
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)
            fh.write("\n")

            # Normalized object 
            if scale is not None:
                fh.write('Scale '+str(scale))


    def write_obj(self, file, curve_strings, total_curve, vertex_strings, total_v, meta_info, scale=None):
        """ Write to .obj file """
        
        with open(file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write(f"# Vertices: {total_v}\n")
            fh.write(f"# Curves: {total_curve}\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            fh.write("ExtrudeOperation: " + meta_info['set_op']+"\n")
            extrude_string = 'Extrude '
            for value in meta_info['extrude_value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")
        
            #Write refe plane transformation 
            p_orig = self.parse3d(meta_info['t_orig'])
            x_axis = self.parse3d(meta_info['t_x'])
            y_axis = self.parse3d(meta_info['t_y'])
            z_axis = self.parse3d(meta_info['t_z'])
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)
            fh.write("\n")

            # Normalized object 
            if scale is not None:
                fh.write('Scale '+str(scale))


    def parse_file(self, scale=1.0):
        """ 
        Parse obj file
        Return
            vertex 2D location numpy
            curve list (geometry class)
            extrude parameters
        """ 
       
        assert self.pathname is not None, "File is None"
        assert self.pathname.exists(), "No such file"

        # Parse file 
        vertex_list = []
        
        # Read vertice
        with open(self.pathname) as obj_file:
            for line in obj_file:
                tokens = line.split()
                if not tokens:
                    continue
                line_type = tokens[0]
                # Vertex
                if line_type == "v":
                    vertex_list.append([float(x) for x in tokens[1:]])
        vertices = np.array(vertex_list, dtype=np.float64) * scale

        # Read curves
        faces = []
        loops = []
        loop = []
        
        # Read in all lines
        lines = []
        with open(self.pathname) as obj_file:
            for line in obj_file:
                lines.append(line)

        # Parse all lines
        faces = []
        for str_idx, line in enumerate(lines):
            tokens = line.split()
            if not tokens:
                continue
            line_type = tokens[0]

            # Start of a new face 
            if line_type == "face":
                faces.append(self.read_face(lines, str_idx+1, vertices))

            # Read meta data
            meta_data = line.strip('# ').strip(' \n').split(' ')
            meta_name = meta_data[0]
        
            if meta_name == 'Extrude':
                extrude_values = [float(x) for x in meta_data[1:]]
                extrude_values = [x*scale for x in extrude_values]
            elif meta_name == 'T_origin':
                t_orig = [float(x) for x in meta_data[1:]] 
                t_orig = [x*scale for x in t_orig] 
            elif meta_name == 'T_xaxis':
                t_x = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_yaxis':
                t_y = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_zaxis':
                t_z = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'ExtrudeOperation:':
                set_op = meta_data[1]

        meta_info = {'extrude_value': extrude_values,
                     'set_op': set_op,
                     't_orig': t_orig,
                     't_x': t_x,
                     't_y': t_y,
                     't_z': t_z,
                    }

        return vertices, faces, meta_info 


    def read_face(self, lines, str_idx, vertices):
        loops = []
        loop = []
        for line in lines[str_idx:]:
            tokens = line.split()
            if not tokens:
                continue
            line_type = tokens[0]

            if line_type == 'face':
                break

            # Start of a new loop 
            if line_type == "out" or line_type == "in":
                if len(loop) > 0:
                    loops.append(loop)
                loop = []
                is_outer = (line_type == 'out')

            # Line
            if line_type == 'l':
                c_tok = tokens[1:]
                curve = Line([int(c_tok[0]), int(c_tok[1])], vertices, is_outer=is_outer)
                loop.append(curve)

            # Arc 
            if line_type == 'a':
                c_tok = tokens[1:]
                curve = Arc([int(c_tok[0]), int(c_tok[1]), int(c_tok[2]), int(c_tok[3])], vertices, is_outer=is_outer)
                loop.append(curve)

            # Circle 
            if line_type == 'c':
                c_tok = tokens[1:]
                curve = Circle([int(c_tok[0]), int(c_tok[1])], vertices, is_outer=is_outer)
                loop.append(curve)

        loops.append(loop)
        return loops
