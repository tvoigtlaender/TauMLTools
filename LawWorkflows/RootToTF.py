import law
import os
import sys
import shutil

from hydra import initialize, compose
from .framework import Task, HTCondorWorkflow, startup_time
from omegaconf import OmegaConf
sys.path.append(os.environ['ANALYSIS_PATH']+'/Preprocessing/root2tf/')
import luigi
import math

law.contrib.load("wlcg")

# Collect all files in the local directory recursively
def collect_files(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    print("file_paths", file_paths)
    return file_paths

# Copy each file to the remote target
def copy_files_to_remote(local_dir, remote_dir):
    print("local_dir, remote_dir", local_dir, remote_dir)
    file_paths = collect_files(local_dir.path)
    print("file_paths", file_paths)

    for local_file_path in file_paths:
        # Get the relative path to create the same structure on the remote side
        rel_path = os.path.relpath(local_file_path, local_dir.path)
        remote_file_target = remote_dir.child(rel_path, type="f")

        # Make sure the remote directory exists
        remote_file_target.parent.touch()

        # Copy the file
        local_file_target = local_dir.child(rel_path, type="f")
        print("COPY", local_file_target, remote_file_target)
        remote_file_target.copy_from_local(local_file_target)
        print(f"Copied {local_file_path} to {remote_file_target.uri()}")

class RootToTF(Task, HTCondorWorkflow, law.LocalWorkflow):
    # class RootToTF(Task, law.LocalWorkflow):
    ## '_' will be converted to '-' for the shell command invocation
    cfg           = luigi.Parameter(description='location of the input yaml configuration file')
    #files_per_job = luigi.IntParameter(default=1, description='number of files to run a single job.')
    n_jobs        = luigi.IntParameter(default=0, description='number of jobs to run. Together with --files-per-job determines the total number of files processed. Default=0 run on all files.')
    dataset_type  = luigi.Parameter(description="which samples to read (train/validation/test)")
    # output_path   = luigi.Parameter(description="output path. Overrides 'path_to_dataset' in the cfg")

    working_dir  = luigi.Parameter(description = 'Path to the working directory.')
    # data_dir  = luigi.Parameter(description = 'Path to the data directory.')
    evictable  = luigi.Parameter(default = "False", description = 'Can job be evicted without breaking?')
    num_CPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested CPU.')
    num_GPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested GPU.')
    accounting_group   = luigi.Parameter(default = "None", significant = False, description = 'Accounting used for TOpAS.')
    cuda_memory  = luigi.Parameter(default = "None", significant = False, description = 'Amount of necessary device memory.')
    requirements = luigi.Parameter(default="None", significant = False, description = 'HTCondor requirements')
    max_disk  = luigi.Parameter(default = 'None', significant = False, description = 'maximum scratch space usage')
    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime")
    max_memory  = luigi.Parameter(default = '2000', significant = False, description = 'maximum RAM usage')
    docker_image = luigi.Parameter(default='None', significant=False, description='Used docker image')

    comp_facility = luigi.Parameter(default = 'ETP', 
                                    description = 'Computing facility for specific setups e.g: desy-naf, lxplus')

    # Redirect location of job files to <local_path>/"files"/...
    def htcondor_create_job_file_factory(self):
        jobdir = self.local_path("files")
        os.makedirs(jobdir, exist_ok=True)
        factory = super(HTCondorWorkflow, self).htcondor_create_job_file_factory(
            dir=jobdir,
            mkdtemp=False,
        )
        return factory
    
    def htcondor_job_config(self, config, job_num, branches):
        config.custom_content = []
        main_dir = os.getenv("ANALYSIS_PATH")
        report_dir = str(self.htcondor_output_directory().path)

        err_dir = '/'.join([report_dir, 'errors'])
        out_dir = '/'.join([report_dir, 'outputs'])
        log_dir = '/'.join([report_dir, 'logs'])

        if not os.path.exists(err_dir): os.makedirs(err_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_path"] = main_dir

        full_req = self.requirements
        if (self.num_GPUs != "None"):
            if self.comp_facility == "TOpAS":
                config.custom_content.append(('request_gpus', self.num_GPUs))
            else:
                config.custom_content.append(('Request_GPUs', self.num_GPUs))
            if self.cuda_memory != "None":
                full_req += " && (GlobalMemoryMb > {})".format(str(self.cuda_memory))
        if full_req != "None":
            config.custom_content.append(("requirements", full_req))
        if self.comp_facility=="desy-naf":
            config.custom_content.append(("+RequestRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('RequestMemory', '{}'.format(self.max_memory)))
        elif self.comp_facility=="lxplus":
            config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('request_memory', '{}'.format(self.max_memory)))
        elif self.comp_facility == "ETP":
            # Use proxy file located in $X509_USER_PROXY or /tmp/x509up_u$(id) if empty
            htcondor_user_proxy = law.wlcg.get_vomsproxy_file()
            # config.render_variables["data_dir"] = self.data_dir
            config.render_variables["comp_facility"] = self.comp_facility
            # config.render_variables["max_files"] = self.max_files
            config.custom_content.append(("x509userproxy", htcondor_user_proxy))
            config.custom_content.append(('+RemoteJob', 'True'))
            config.custom_content.append(("+RequestWalltime", int(math.floor(self.max_runtime * 3600)) - 1))
            for i in ["num_CPUs", "max_memory", "max_disk", "accounting_group", "docker_image"]:
                if getattr(self, i) == "None":
                    raise Exception('ETP requires a value for {}.'.format(i))
            config.custom_content.append(('request_cpus', self.num_CPUs))
            config.custom_content.append(('RequestMemory', self.max_memory))
            # config.custom_content.append(('Request_GPUMemoryMB', '0'))
            if self.evictable:
                config.custom_content.append(('+evictable', self.evictable))
            config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
            config.custom_content.append(('accounting_group', self.accounting_group))
            config.custom_content.append(("universe", "docker"))
            config.custom_content.append(("docker_image", self.docker_image))
            # config.custom_content.append(("docker_network_type", "host")) # Fix for xrootd issues on TOpAS, nor clear what the issue is, but it works with the host network
            tarball_dir = os.path.abspath(f"tarballs/{self.version}")
            tarball_local = law.LocalFileTarget(
                os.path.join(
                    tarball_dir,
                    self.__class__.__name__,
                    "TauMLTools.tar.gz",
                )
            )
            if not tarball_local.exists():
                tarball_local.parent.touch()
                # os.system("mkdir -p tarballs")
                # os.system("rm tarballs/TauMLTools.tar.gz")
                os.system(f'tar --exclude={{"TauMLTools/tarballs","TauMLTools/soft","TauMLTools/data","__pycache__"}} -czf {tarball_local.path}  ../TauMLTools')
            config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
            # config.input_files["copy_script"] = law.JobInputFile("copy_in.sh", render=False)
            # config.output_files.append("mlruns.tar.gz")
        else:
            raise Exception('no specific setups for {self.comp_facility} computing facility')

        if self.comp_facility != "ETP":
            config.custom_content.append(("getenv", "true"))
        config.render_variables["environment"] = self.environment
        config.render_variables["LOCAL_TIMESTAMP"] = startup_time
        config.custom_content.append(('JobBatchName'  , self.batch_name))
        config.custom_content.append(("error" , '/'.join([err_dir, 'err_{}.txt'.format(job_num)])))
        config.custom_content.append(("output", '/'.join([out_dir, 'out_{}.txt'.format(job_num)])))
        config.custom_content.append(("log"   , '/'.join([log_dir, 'log_{}.txt'.format(job_num)])))
        # config.custom_content.append(("stream_error", "True"))
        # config.custom_content.append(("stream_output", "True"))
        return config
    
    
    def __init__(self, *args, **kwargs):
        ''' run the conversion of .root files to tensorflow datasets
        '''
        super(RootToTF, self).__init__(*args, **kwargs)
        # the task is re-init on the condor node, so os.path.abspath would refer to the condor node root directory
        # re-instantiating luigi parameters bypasses this and allows to pass local paths to the condor job
        rel_cfg = os.path.relpath(self.cfg)
        # print("HERE", rel_cfg)
        # print("HERE", os.path.basename(rel_cfg))
        # print("HERE", os.path.dirname(rel_cfg))
        with initialize(config_path=os.path.dirname(rel_cfg)):
            self.cfg_dict = compose(config_name=os.path.basename(rel_cfg))
        # print(self.cfg_dict)
        input_data  = OmegaConf.to_object(self.cfg_dict['input_data'])
        self.dataset_cfg = input_data[self.dataset_type]
        #    self.output_path = os.path.abspath(self.cfg_dict['path_to_dataset'])
        
        # self.output_path = os.path.abspath(self.output_path)
        # if not os.path.exists(self.output_path):
        #     os.makedirs(self.output_path)
        # self.cfg_dict['path_to_dataset'] = self.output_path

    def move(self, src, dest):
        #if os.path.exists(dest):
        #  if os.path.isdir(dest): shutil.rmtree(dest)
        #  else: os.remove(dest)
        shutil.move(src, dest)

    def create_branch_map(self):
        from create_dataset import fetch_file_list
        _files  = self.dataset_cfg.pop('files')
        files   = sorted([f if f.startswith('root://') else os.path.abspath(f) for f in _files ])
        files   = fetch_file_list(files)
        assert len(files), "Input file list is empty: {}".format(_files)
        branch_map = {i: j for i,j in enumerate(files)}
        # print("HERE", branch_map)
        return branch_map



    # # Define output targets. Task is considerd complete if all targets are present.
    # def output(self):
    #     identifier, training_class = self.branch_data["datashard_information"]
    #     file_ = self.files_template.format(
    #         identifier=identifier,
    #         training_class=training_class,
    #         fold=self.branch_data["fold"],
    #     )
    #     # target = self.local_target(file_)
    #     target = self.remote_target(file_)
    #     target.parent.touch()
    #     return target


    def output(self):  
        file_path = self.branch_data
        # print(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # print(file_name)
        # print(self.output_path, "STOP", self.dataset_type, "STOP", file_name)
        # output_dir = os.path.join(self.output_path, self.dataset_type, file_name)
        # print(output_dir)
        # # output_target = self.remote_directory_target(output_dir)
        output_target = self.remote_directory_target(f"{self.dataset_type}/{file_name}")
        output_target.parent.touch()
        # print(output_target)
        # print(output_target.path)
        # print(output_target.exists)
        # print(output_target.parent.parent.path)
        # exit(0)
        return output_target

    def run(self):
        from create_dataset import process_files as run_job
        file_path = self.branch_data
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        temp_output_folder = os.path.abspath('./temp/{}'.format(file_name))
        self.cfg_dict['path_to_dataset'] = temp_output_folder
        print(f"file_path = {file_path}")
        result = run_job( 
            cfg           = self.cfg_dict     ,
            dataset_type  = self.dataset_type ,
            files         = file_path  ,
            dataset_cfg   = self.dataset_cfg  ,
        )
        if not result:
            raise Exception('job {} failed'.format(self.branch))
        else:
            copy_files_to_remote(law.LocalDirectoryTarget(temp_output_folder), self.output().parent.parent)
            # self.output().copy_from_local(law.LocalDirectoryTarget(temp_output_folder))
            print('Output files moved to {}'.format(self.output().path))
