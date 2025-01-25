folder_names = ['pcfea']
pair_names = ['ms','s+s','ss+','sm','s+m','ms+']
dataset_dict = {'s+s':['shapenet', 'scannet'],'ss+':['scannet','shapenet'],
'sm':['scannet', 'modelnet'],'ms':['modelnet','scannet'],'s+m':['shapenet','modelnet'],
'ms+':['modelnet','shapenet']}

file_names = ['train_PCFEA_cls.py']

for j, folder_name in enumerate(folder_names):
    for pair in pair_names:
        for seed in range(1,4):
            file_name = folder_name + '/'+pair+str(seed)+'.sh'
            with open(file_name, 'w') as f:
                
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --partition=gpu\n')
                f.write('#SBATCH --nodes=1\n')
                f.write('#SBATCH --cpus-per-task=1\n')
                f.write('#SBATCH --ntasks-per-node=4\n')
                f.write('#SBATCH --mem-per-cpu=32GB\n')
                f.write('#SBATCH --time=20:00:00\n')
                f.write('#SBATCH --gres=gpu:1\n')
                f.write('#SBATCH --output='+pair+str(seed)+'{}.out'.format(folder_name)+'\n')
                f.write('\n\n\n\n')
                f.write('eval "$(conda shell.bash hook)"\n')
                f.write('conda activate pyCLGL\n')
                f.write('\n\n\n\n')

                f.write('python /scratch1/mengxiwu/PCFEA/'+file_names[j] + ' --dataroot /scratch1/mengxiwu/PCFEA/data/PointDA/'+ 
                            ' --src_dataset '+ dataset_dict[pair][0] + ' --trgt_dataset '+ dataset_dict[pair][1]
                            + ' --seed '+ str(seed)  
                            + ' --model_type '+folder_name+'\n') 

         
f.close()

for j, folder_name in enumerate(folder_names):
    with open(folder_name+'.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --cpus-per-task=1\n')
        f.write('#SBATCH --ntasks-per-node=1\n')
        f.write('#SBATCH --mem-per-cpu=2GB\n')
        f.write('#SBATCH --time=00:5:00\n')
        f.write('#SBATCH --output={}.out\n'.format(folder_name))
        f.write('\n\n')
        for pair in pair_names:
            for seed in [1,2,3]:
                file_name = folder_name + '/'+pair+str(seed)+'.sh'
                f.write('sbatch '+file_name+'\n')
f.close()