from datetime import datetime
from os import mkdir
import os
from copy import copy,deepcopy
import time
import glob
import subprocess

def getDateString():
	return(datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))


def makeDir(dir_name):
	# Even if this is in a library dir, it should make the dir
	# in the script that called it.
	mkdir(dir_name)
	return(dir_name)


def makeDateDir(base_dir='.'):
	# Just creates a dir with the current date for its name
	ds = getDateString()
	full_dir = combineDirAndFile(base_dir, ds)
	makeDir(full_dir)
	return(full_dir)


def makeLabelDateDir(label, base_dir='.'):
	# You give it a label, and it creates the dir label_datestring
	dir_name = label + '_' + getDateString()
	full_dir = combineDirAndFile(base_dir, dir_name)
	makeDir(full_dir)
	return(full_dir)


def combineDirAndFile(dir, file):
	# Adds the file to the end of dir, adding a slash in between if needed.
	return(addTrailingSlashIfNeeded(dir) + file)


def dictPrettyPrint(in_dict):

	# Formats a dict into a nice string with each k,v entry on a new line,
	# and prints it.
	dict_str = '{\n'

	for k,v in in_dict.items():
		dict_str += '\t{} : {}\n'.format(k, v)

	dict_str += '\n}\n'
	print(dict_str)




def dictToStringList(dict):
	pd_copy = copy(dict)
	for k,v in pd_copy.items():
		if type(v).__name__ == 'float':
			if abs(v)>10**-4:
				pd_copy[k] = '{:.5f}'.format(v)
			else:
				pd_copy[k] = '{:.2E}'.format(v)

	params = [str(k)+'='+str(v) for k,v in pd_copy.items() if v is not None]
	return(params)



def paramDictToFnameStr(param_dict):
	# Creates a string that can be used as an fname, separated by
	# underscores. If a param has the value None, it isn't included.
	params = dictToStringList(param_dict)
	return('_'.join(params))

def paramDictToLabelStr(param_dict):
	# Creates a string that can be used as an fname, separated by
	# ', '. If a param has the value None, it isn't included.
	params = dictToStringList(param_dict)
	return(', '.join(params))


def listToFname(list):
	return('_'.join(list))


def parseSingleAndListParams(param_dict, exclude_list):

	# This is useful for if you want to do multiple runs, varying one or
	# several parameters at once. exclude_list are ones you don't want to
	# include in the parameters in the tuple.

	# It returns a list of the parameters that are varied,
	# and a list of dictionaries that can be directly passed to a function, where
	# each one has a different set of the varied params.
	#
	# You should pass the args where if you don't want to vary an arg, it's just normal
	# my_arg = 5, but if you do want to vary it, you pass it a list of the vary values, like
	# my_arg = [1, 5, 8]. If you want to vary two at the same time, you pass them both as separate
	# lists, and it will match them up, but they need to be the same size.

	# list_params is just a list of the params that were passed as a list, that we'll vary.
	list_params = []
	# single_params is a dict of the params that aren't varied and will have the same vals in each
	# separate run.
	single_params = {}
	# ziplist is a list of the lists for the params that are varied. So if there are two varied
	# args, each length 3, it will take these, and then below zip them to create a list of pairs.
	# arg1=[1,2,3], arg2=[2,4,8] -> ziplist=[arg1,arg2] -> param_tups=[(1,2),(2,4),(3,8)]
	ziplist = []


	for k,v in param_dict.items():
		if type(v).__name__ == 'list':
			list_params.append(k)
			ziplist.append(v)
		else:
			if k not in exclude_list:
				single_params[k] = v

	param_tups = list(zip(*ziplist))

	vary_param_dicts = []
	vary_param_tups = []
	for tup in param_tups:
		temp_dict = dict(zip(list_params,tup))
		temp_kw = {**single_params, **temp_dict}
		vary_param_tups.append(temp_dict)
		vary_param_dicts.append(temp_kw)

	# list_params: just a list of the names of the varied ones.
	# vary_param_dicts: a list of the dicts that you can pass to each iteration, which includes the args that don't vary.
	# vary_param_tups: a list of dicts corresponding to vary_param_dicts, of only the values that change.
	return(list_params, vary_param_dicts, vary_param_tups)



def strfdelta(tdelta, fmt):
	d = {"days": tdelta.days}
	d["hours"], rem = divmod(tdelta.seconds, 3600)
	d["minutes"], d["seconds"] = divmod(rem, 60)
	return fmt.format(**d)


def getCurTimeObj():
	return(datetime.now())


def getTimeDiffNum(start_time_obj):

	diff = datetime.timestamp(datetime.now()) - datetime.timestamp(start_time_obj)
	return(diff)


def getTimeDiffObj(start_time_obj):
	#Gets the time diff in a nice format from the start_time_obj.
	diff = datetime.now() - start_time_obj
	return(diff)


def getTimeDiffStr(start_time_obj):
	#Gets the time diff in a nice format from the start_time_obj.
	diff = getTimeDiffObj(start_time_obj)

	return(strfdelta(diff,'{hours} hrs, {minutes} mins, {seconds} s'))


def writeDictToFile(dict, fname):
	# You have to copy it here, otherwise it'll actually overwrite the values in the dict
	# you passed.
	my_dict = copy(dict)
	f = open(fname,'w+')
	for k,v in my_dict.items():
		if type(v).__name__ == 'float':
			if abs(v)>10**-4:
				my_dict[k] = '{:.5f}'.format(v)
			else:
				my_dict[k] = '{:.2E}'.format(v)
		f.write('{} = {}\n'.format(k, my_dict[k]))

	f.close()


def readFileToDict(fname):
	d = {}
	with open(fname) as f:
		for line in f:
			(key, val) = line.split(' = ')
			val = val.strip('\n')
			#This is to handle the fact that everything gets read in
			#as a string, but some stuff you probably want to be floats.
			try:
				val = float(val)
			except:
				val = str(val)

			d[key] = val


	return(d)


def dirFromFullPath(fname):
	# This gives you the path, stripping the local filename, if you pass it
	# a long path + filename.
	parts = fname.split('/')
	last_part = parts[-1]
	path = fname.replace(last_part,'')
	if path == '':
		return('./')
	else:
		return(path)


def fnameFromFullPath(fname):
	# This just gets the local filename if you passed it some huge long name with the path.
	parts = fname.split('/')
	last_part = parts[-1]
	return(last_part)

def stripAnyTrailingSlash(path):
	if path[-1] == '/':
		return(path[:-1])
	else:
		return(path)


def addTrailingSlashIfNeeded(path):
	if path[-1] == '/':
		return(path)
	else:
		return(path + '/')





def gifFromImages(imgs_path, gif_name, ext = '.png', delay=50):


	imgs_path = stripAnyTrailingSlash(imgs_path)
	file_list = glob.glob(imgs_path + '/' + '*' + ext) # Get all the pngs in the current directory
	#print(file_list)
	#print([fnameFromFullPath(x).split('.png')[0] for x in file_list])
	#list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0]))
	list.sort(file_list, key=lambda x: int(fnameFromFullPath(x).split(ext)[0]))
	#list.sort(file_list) # Sort the images by #, this may need to be tweaked for your use case
	#print(file_list)
	assert len(file_list) < 300, 'Too many files ({}), will probably crash convert command.'.format(len(file_list))

	output_fname = '{}/{}.gif'.format(imgs_path, gif_name)

	check_call_arglist = ['convert'] + ['-delay', str(delay)] + file_list + [output_fname]
	#print(check_call_arglist)
	print('Calling convert command to create gif...')
	subprocess.check_call(check_call_arglist)
	print('done.')
	return(output_fname)
	# older method:

	'''with open('image_list.txt', 'w') as file:
	    for item in file_list:
	        file.write("%s\n" % item)

	os.system('convert @image_list.txt {}/{}.gif'.format(imgs_path,gif_name)) # On windows convert is 'magick'
	'''

#
