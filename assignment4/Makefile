.PHONY: runcl
runcl:
	sbatch --gres=gpu:1 --time=8 --wrap="python histCL.py"
.PHONY: runcuda
runcuda:
	sbatch --gres=gpu:1 --time=8 --wrap="python histCuda.py"
.PHONY: clean
clean:
	rm -f *.o a.out core slurm*
.PHONY:all
all: clean runcuda
