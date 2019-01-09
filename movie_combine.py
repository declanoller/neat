
from moviepy.editor import VideoFileClip, clips_array, vfx
import argparse
import FileSystemTools as fst
import glob
import subprocess
import os


# arguments to be read in via CLI
parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--grid_size', default='0')
parser.add_argument('--file_type', default='mp4')
parser.add_argument('--gif', action='store_true', default=False)
args = parser.parse_args()

# get the files with the video clip extension type
file_list = glob.glob(fst.addTrailingSlashIfNeeded(args.path) + '*' + args.file_type)
print('{} files of type {} found'.format(len(file_list), args.file_type))

# make sure you've passed a grid size argument
assert args.grid_size != '0', 'need to provide a grid_size arg of form <height int>x<width int>'

try:
    grid_dims = [int(y) for y in args.grid_size.split('x')]
    grid_height, grid_width = grid_dims[0], grid_dims[1]
    N_movie_panels = grid_height*grid_width
    print('need {} movie files for a grid of size {}'.format(N_movie_panels, args.grid_size))
except:
    print('something wrong with grid_size argument, should be of form 5x8 (or similar)')
    exit()

# take only the first N video files, no choosing process. It will use ones created
# from running this program previously if they're there, so be careful.
files_used = file_list[:N_movie_panels]

clip_list = []
clip_matrix = []

# create a list of the video file clip objects, with a small margin around each
for f in files_used:
    clip1 = VideoFileClip(f).margin(10)
    #clip1 = clip1.resize(0.50)
    clip_list.append(clip1)

# put them into a list of lists, ie, a matrix, in the shape you want them to finally be
for y in range(grid_height):
    temp_list = []
    for x in range(grid_width):
        temp_list.append(clip_list[y*grid_width + x])

    clip_matrix.append(temp_list)

print('size of clip_matrix:', len(clip_matrix), len(clip_matrix[0]))
final_clip = clips_array(clip_matrix) # put the clips side by side

# fname stuff
dt_string = fst.getDateString()
base_fname = 'COMBINED_{}_{}'.format(args.grid_size, dt_string)
movie_output_fname = fst.combineDirAndFile(args.path, '{}.{}'.format(base_fname, args.file_type))

final_clip.write_videofile(movie_output_fname) # create the combined video file!

if args.gif:

    px_size = 1260
    fps = 30

    gif_output_fname = fst.combineDirAndFile(args.path, '{}.gif'.format(base_fname))

    palette_fname = 'palette.png'

    create_palette_cmd = 'ffmpeg -y  -i {} -vf fps={},scale={}:-1:flags=lanczos,palettegen {}'.format(movie_output_fname, fps, px_size, palette_fname)
    create_gif_cmd = 'ffmpeg -i {} -i {} -filter_complex "fps={},scale={}:-1:flags=lanczos[x];[x][1:v]paletteuse" {}'.format(movie_output_fname, palette_fname, fps, px_size, gif_output_fname)

    os.system(create_palette_cmd)
    os.system(create_gif_cmd)

    remove_palette_cmd = f'rm {palette_fname}'
    remove_movie_cmd = f'rm {movie_output_fname}'

    os.system(remove_palette_cmd)
    os.system(remove_movie_cmd)



exit()

################# scrap

'''
parser.add_argument('--gif', action='store_true', default=False)
parser.add_argument('--refreshrate', default=2, type=float)


clip1 = VideoFileClip("myvideo.mp4").margin(10) # add 10px contour
clip2 = clip1.fx( vfx.mirror_x)
clip3 = clip1.fx( vfx.mirror_y)
clip4 = clip1.resize(0.60) # downsize 60%
final_clip = clips_array([[clip1, clip2],
                          [clip3, clip4]])
final_clip.resize(width=480).write_videofile("my_stack.mp4")
'''







#
