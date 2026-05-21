# Totient Cluster Notes

Totient uses TORQUE/PBS and older RedHat infrastructure. It is useful for CPU
batch timing runs, but it is not the first choice for this Python/PyTorch
pipeline compared with G2/Lisbeth-style CPU resources.

Use this only after testing the environment manually:

```bash
module avail
module load anaconda/4.4.0
```

The head node should not run computationally intensive jobs. Submit PBS jobs
with `qsub`.
