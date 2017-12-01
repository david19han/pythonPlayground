.PHONY: runcl
runcl:
	sbatch --gres=gpu:1 --time=8 --wrap="python histCL.py"
.PHONY: runcuda
runcuda:
	sbatch --gres=gpu:1 --time=8 --wrap="python histCuda.py"
.PHONY: tempcl
tempcl:
	sbatch --gres=gpu:1 --time=8 --wrap="python tempCl.py"
.PHONY: tempcuda
tempcuda:
	sbatch --gres=gpu:1 --time=8 --wrap="python tempCuda.py"
.PHONY: clean
clean:
	rm -f *.o a.out core slurm*
.PHONY:all
all: clean runcuda
