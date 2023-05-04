import multiprocessing as mp
import numpy as np
import subprocess
import argparse
import shutil
import gzip
import os

# https://github.com/quqixun/BrainPrep/blob/master/src/registration.py#L16
# register (align) brain to standard template
def registration(src_path, dst_path, ref_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)
    return

# https://github.com/quqixun/BrainPrep/blob/master/src/registration.py#L26
# reorient brain scan before registering to template
def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)
    return

# https://github.com/quqixun/BrainPrep/blob/master/src/skull_stripping.py#L13
# use BET to skull strip brain
def bet(src_path, dst_path, frac="0.5"):
    command = ["bet", src_path, dst_path, "-R", "-f", frac, "-g", "0"]
    subprocess.call(command)
    return

# preprocess a .nii at `src_path` and write it to `dst_path`
# use BET to skull strip the brain, include brain matter that was included with a confidence of `strip_confidence`
def process_img(src_path, dst_path, strip_confidence = "0.4"):
	if os.path.exists(dst_path):
		return True

	parent_dir = os.path.dirname(dst_path)
	if len(parent_dir) > 0 and not os.path.exists(parent_dir):
		os.makedirs(parent_dir)

	# some of the FSL tools just output the preprocessed file as a .nii.gz
	tmp_path = dst_path + "_tmp.nii.gz"
	success = False

	try:
		orient2std(src_path, tmp_path)
		registration(tmp_path, tmp_path, "MNI152_T1_1mm.nii.gz") # register to 1mm template, which has dimensions of (182, 218, 182)
		bet(tmp_path, tmp_path, strip_confidence)

		success = True
	except RuntimeError:
		print(f"Failed to process {src_path}")

	# copy created file to dst_path and remove any temporarily created files
	if success:
		with gzip.open(tmp_path, 'rb') as f_in:
			with open(dst_path, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

	if os.path.exists(tmp_path):
		os.remove(tmp_path)

	return success

# Preprocesses a set of .nii files. Creates new directory with '_FSL' appended to name for preprocessed files
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help = "directory to preprocess")
	args = parser.parse_args()

	paths = []
	search_path = os.path.abspath(os.path.join(os.getcwd(), args.dir))
	idx = len(search_path)

	# build list of files to preprocess.
	for (root, dirs, files) in os.walk(search_path):
		for file in files:
			if file.endswith('.nii'):
				src = os.path.join(root, file)
				dst = src[:idx] + "_FSL" + src[idx:]
				
				paths.append((src, dst))
	
	# Multiprocessing make things fast :)
	print(f"Processing {len(paths)} files")

	cnt, total = 0, 0
	for src, dst in paths:
		if not os.path.exists(dst):
			cnt += 1

		total += 1

	print(cnt, total)
	'''
	with mp.Pool(processes = mp.cpu_count()) as pool:
		results = pool.starmap(process_img, paths)

	# print out scans that failed to process, doesn't really work properly but I haven't really needed it
	# If preprocessing fails, the child process usually prints an error and exits/crashes so the preprocessed file just wont be created
	if False in results:
		paths = np.array(paths)
		results = np.array(results)

		for src, dst in paths[np.where(results == False)[0]]:
			print(f"Failed {src}")

	print("Finished Preprocessing Files")
	'''

if __name__ == '__main__':
	main()