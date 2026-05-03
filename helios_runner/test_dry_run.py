from rfpose_helios.submit import HeliosJobSpec, render_sbatch
spec=HeliosJobSpec(job_id='dryrun',dataset_version='rfpose-test',train_config='rf_worldpose_lora',account='TEST-gpu-gh200')
text=render_sbatch(spec)
assert 'plgrid-gpu-gh200' in text and 'rfpose-test' in text and 'sbatch' not in text.lower().split('\n')[0]
print('helios dry-run render ok')
