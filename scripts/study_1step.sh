#!/bin/bash
set -Ceux

if [ ! $# -eq 1 ]
then
  echo "Usage: $0 fistr_input_directory"
  exit 0
fi
reference_directory=${1%/}

res_file="mesh_vis_psf.0070.inp"
memory=250
cpu=1


# MUMPS
mumps_directory=${reference_directory}_1step
if [ ! -d $mumps_directory ]
then
  cp -r $reference_directory $mumps_directory
fi

backup_cnt=${mumps_directory}/mesh.cnt.bak
if [ ! -f ${backup_cnt} ]
then
  mv ${mumps_directory}/mesh.cnt $backup_cnt
  sed 's/0.01,1.0/1.0,1.0/' $backup_cnt > ${mumps_directory}/mesh.cnt
fi

if [ ! -f ${mumps_directory}/${res_file} ]
then
  pushd $mumps_directory
  qexe -c $cpu -m $memory fistr1
  popd
fi


# CG
cg_directory=${reference_directory}_1step_cg
if [ ! -d $cg_directory ]
then
  cp -r $mumps_directory $cg_directory
fi

cg_backup_cnt=${cg_directory}/mesh.cnt.bak.cg
if [ ! -f ${cg_backup_cnt} ]
then
  mv ${cg_directory}/mesh.cnt $cg_backup_cnt
   sed 's/METHOD=MUMPS/METHOD=CG/' $cg_backup_cnt > ${cg_directory}/mesh.cnt
fi

if [ ! -f ${cg_directory}/${res_file} ]
then
  pushd $cg_directory
  qexe -c $cpu -m $memory fistr1
  popd
fi
