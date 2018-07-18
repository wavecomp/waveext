# wf_env.sh
# 
# Copyright (C) Wave Computing 2018
# All rights reserved.
# 
# PURPOSE
# 	Set all required environment variables for the waveflow project.
# 
# USE
#	From a bash shell, type: ". path/to/repo/wf_env.sh"	
#
# Author          : Ken Shiring
# Created On      : 03/16/2018
# 

# First use any arg. If not defined, default to the directory prefix
# of this script.
repo_rel=${1:-$(dirname "$0")}
echo "repo_rel: $repo_rel"

export WF_REPO_DIR=$(readlink -e $repo_rel)

echo "Using repository: $WF_REPO_DIR"

export CC=clang
export CXX=clang++
export PYTHONPATH=${PYTHONPATH}:${WF_REPO_DIR}


# Helper functions for building
function wfbuild_type()
{
	case "${1}" in
		'')	echo "Release" ;;
		r)	echo "Release" ;;
		d)	echo "Debug" ;;
		*)
			echo "Don't know about build ${1}, using Debug"
			echo "Debug"
			;;
	esac
}

function wfget_omp()
{
	case "${1}" in
		'')	echo "False" ;;
		omp)	echo "True" ;;
		*)
			echo "Don't know about param ${1}, going w/o OPENMP"
			echo "False"
			;;
	esac
}

function wfbuild()
{
	cores_for_build=${NUM_CORES:-2}
	build_type=$(wfbuild_type "$1")
	omp=$(wfget_omp "$2")
	# echo "Got build: $build_type"
	# echo "OPENMP: $omp"

	mkdir -p $WF_REPO_DIR/build
	cd $WF_REPO_DIR/build
	echo "Building waveflow in $build_type mode and OPENMP=$omp"
	echo ""
	cmake -DCMAKE_BUILD_TYPE=$build_type -DOPENMP_OPTIMIZATION=$omp ..
	make -j $cores_for_build
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "make failed"
		return $rc
	fi
	cd ..
}
