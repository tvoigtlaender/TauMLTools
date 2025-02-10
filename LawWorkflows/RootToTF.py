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
    return file_paths

# Copy each file to the remote target
def copy_files_to_remote(local_dir, remote_dir):
    file_paths = collect_files(local_dir.path)

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
    n_jobs        = luigi.IntParameter(default=0, description='number of jobs to run. Together with --files-per-job determines the total number of files processed. Default=0 run on all files.')
    dataset_type  = luigi.Parameter(description="which samples to read (train/validation/test)")
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
        config.render_variables["copy_in"] = "False"

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
            config.render_variables["comp_facility"] = self.comp_facility
            config.custom_content.append(("x509userproxy", htcondor_user_proxy))
            config.custom_content.append(('+RemoteJob', 'True'))
            config.custom_content.append(("+RequestWalltime", int(math.floor(self.max_runtime * 3600)) - 1))
            for i in ["num_CPUs", "max_memory", "max_disk", "accounting_group", "docker_image"]:
                if getattr(self, i) == "None":
                    raise Exception('ETP requires a value for {}.'.format(i))
            config.custom_content.append(('request_cpus', self.num_CPUs))
            config.custom_content.append(('RequestMemory', self.max_memory))
            if self.evictable:
                config.custom_content.append(('+evictable', self.evictable))
            config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
            config.custom_content.append(('accounting_group', self.accounting_group))
            config.custom_content.append(("universe", "docker"))
            config.custom_content.append(("docker_image", self.docker_image))
            tarball_dir = os.path.abspath(f"{main_dir}/tarballs/{self.version}")
            tarball_local = law.LocalFileTarget(
                os.path.join(
                    tarball_dir,
                    self.__class__.__name__,
                    "TauMLTools.tar.gz",
                )
            )
            if not tarball_local.exists():
                tarball_local.parent.touch()
                excludes = ["./.[^.]*", "./Analysis", "./Production", "./Evaluation", "./Core", "./Training", "./RunKit", "./soft", "./data", "./tarballs", "*/outputs", "*/mlruns", "__pycache__"]
                exclude_str = " ".join([f"--exclude={ex}" for ex in excludes])
                os.system(f'tar {exclude_str} -czf {tarball_local.path}  .')
                tarball_local.parent.touch()
            config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
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
        rel_cfg = os.path.relpath(self.cfg, f"{os.getenv('ANALYSIS_PATH')}/LawWorkflows")
        with initialize(config_path=os.path.dirname(rel_cfg)):
            self.cfg_dict = compose(config_name=os.path.basename(rel_cfg))
        input_data  = OmegaConf.to_object(self.cfg_dict['input_data'])
        self.dataset_cfg = input_data[self.dataset_type]

    def create_branch_map(self):
        from utils.remote_glob import remote_glob
        _files = self.dataset_cfg.pop('files')
        files = []
        for file_path in _files:
            files += remote_glob(file_path)
        assert len(files), "Input file list is empty: {}".format(_files)
        branch_map = {i: j for i,j in enumerate(files)}
        return branch_map


    def output(self):  
        file_path = self.branch_data
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_target = self.remote_directory_target(file_name)
        output_target.parent.touch()
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
            files         = [file_path]  ,
            dataset_cfg   = self.dataset_cfg  ,
        )
        if not result:
            raise Exception('job {} failed'.format(self.branch))
        else:
            copy_files_to_remote(law.LocalDirectoryTarget(temp_output_folder), self.output().parent.parent)
            print('Output files moved to {}'.format(self.output().path))
