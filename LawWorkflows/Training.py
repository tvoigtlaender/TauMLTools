## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import law
import subprocess
import os
import re
import math
import select
from hydra import compose, initialize
from .framework import Task, HTCondorWorkflow, startup_time
import luigi
from mass_copy import mass_copy
from law.util import interruptable_popen
law.contrib.load("wlcg")

class Training(Task, HTCondorWorkflow, law.LocalWorkflow):

    working_dir  = luigi.Parameter(description = 'Path to the working directory.')
    # data_dir_train  = luigi.Parameter(description = 'Path to the data directory of training data.')
    # data_dir_val  = luigi.Parameter(description = 'Path to the data directory of validation data.')
    # max_files_train  = luigi.Parameter(default = "10000000", description = 'Maximum number of files allowed for training')
    # max_files_val  = luigi.Parameter(default = "10000000", description = 'Maximum number of files allowed for validation')
    evictable  = luigi.Parameter(default = "False", description = 'Can job be evicted without breaking?')
    num_CPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested CPU.')
    num_GPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested GPU.')
    accounting_group   = luigi.Parameter(default = "1", significant = False, description = 'Accounting used for ETP.')
    cuda_memory  = luigi.Parameter(default = "None", significant = False, description = 'Amount of necessary device memory.')
    input_cmds   = luigi.Parameter(description = 'Path to the txt file with input commands.')
    requirements = luigi.Parameter(default=f"(OpSysAndVer =?= \"CentOS7\")", significant = False, description = 'HTCondor requirements')
    max_disk  = luigi.Parameter(default = 'None', significant = False, description = 'maximum scratch space usage')
    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime")
    max_memory  = luigi.Parameter(default = '2000', significant = False, description = 'maximum RAM usage')
    docker_image = luigi.Parameter(default='None', significant=False, description='Used docker image')

    comp_facility = luigi.Parameter(default = 'desy-naf', 
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

    def convert_env_to_dict(self, env):
        my_env = {}
        for line in env.splitlines():
            if line.find(" ") < 0:
                try:
                    key, value = line.split("=", 1)
                    my_env[key] = value
                except ValueError:
                    pass
        return my_env

    def set_environment(self, sourcescript, silent=False):
        if not silent:
            print("with source script: {}".format(sourcescript))
        if isinstance(sourcescript, str):
            sourcescript = [sourcescript]
        source_command = [
            "source {};".format(sourcescript) for sourcescript in sourcescript
        ] + ["env"]
        source_command_string = " ".join(source_command)
        code, out, error = interruptable_popen(
            source_command_string,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # rich_console=console
        )
        if code != 0:
            print("source returned non-zero exit status {}".format(code))
            print("Error: {}".format(error))
            raise Exception("source failed")
        my_env = self.convert_env_to_dict(out)
        return my_env

    # Run a bash command
    #   Command can be composed of multiple parts (interpreted as seperated by a space).
    #   A sourcescript can be provided that is called by set_environment the resulting
    #       env is then used for the command
    #   The command is run as if it was called from run_location
    #   With "collect_out" the output of the run command is returned
    def run_command(
        self,
        command=[],
        sourcescript=[],
        run_location=None,
        collect_out=False,
        silent=False,
    ):
        if command:
            if isinstance(command, str):
                command = [command]
            logstring = "Running {}".format(command)
            if run_location:
                logstring += " from {}".format(run_location)
            if not silent:
                print(logstring)
            if sourcescript:
                run_env = self.set_environment(sourcescript, silent)
            else:
                run_env = None
            code, out, error = interruptable_popen(
                " ".join(command),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
                cwd=run_location,
            )
            if not silent:
                print("Output: {}".format(out))
            if not silent or code != 0:
                print("Error: {}".format(error))
            if code != 0:
                print("Error when running {}.".format(list(command)))
                print("Command returned non-zero exit status {}.".format(code))
                raise Exception("{} failed".format(list(command)))
            else:
                if not silent:
                    print("Command successful.")
            if collect_out:
                return out
        else:
            raise Exception("No command provided.")
        
    def run_command_readable(self, command=[], sourcescript=[], run_location=None):
        """
        This can be used, to run a command, where you want to read the output while the command is running.
        redirect both stdout and stderr to the same output.
        """
        if command:
            if isinstance(command, str):
                command = [command]
            if sourcescript:
                run_env = self.set_environment(sourcescript)
            else:
                run_env = None
            logstring = "Running {}".format(command)
            if run_location:
                logstring += " from {}".format(run_location)
            print("--------------------")
            print(logstring)
            try:
                p = subprocess.Popen(
                    " ".join(command),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=run_env,
                    cwd=run_location,
                    encoding="utf-8",
                )
                while True:
                    reads = [p.stdout.fileno(), p.stderr.fileno()]
                    ret = select.select(reads, [], [])

                    for fd in ret[0]:
                        if fd == p.stdout.fileno():
                            read = p.stdout.readline()
                            if read != "\n":
                                print(read.strip())
                        if fd == p.stderr.fileno():
                            read = p.stderr.readline()
                            if read != "\n":
                                print(read.strip())

                    if p.poll() != None:
                        break
                if p.returncode != 0:
                    raise Exception(f"Error when running {command}.")
            except Exception as e:
                raise Exception(f"Error when running {command}.")
        else:
            raise Exception("No command provided.")

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
            if self.comp_facility == "ETP":
                config.custom_content.append(('request_gpus', self.num_GPUs))
            else:
                config.custom_content.append(('Request_GPUs', self.num_GPUs))
            if self.cuda_memory != "None":
                full_req += " && (GlobalMemoryMb > {})".format(str(self.cuda_memory))
        config.custom_content.append(("requirements", full_req))
        if self.comp_facility=="desy-naf":
            config.custom_content.append(("+RequestRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('RequestMemory', '{}'.format(self.max_memory)))
        elif self.comp_facility=="lxplus":
            config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('request_memory', '{}'.format(self.max_memory)))
        elif self.comp_facility == "ETP":
            # Use proxy file located in $X509_USER_PROXY or /tmp/x509up_u$(id) if empty data_dir_val
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
            # config.custom_content.append(('Request_GPUMemoryMB', '0'))
            if self.evictable:
                config.custom_content.append(('+evictable', self.evictable))
            config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
            config.custom_content.append(('accounting_group', self.accounting_group))
            config.custom_content.append(("universe", "docker"))
            config.custom_content.append(("docker_image", self.docker_image))
            # config.custom_content.append(("docker_network_type", "host")) # Fix for xrootd issues on ETP, nor clear what the issue is, but it works with the host network
            
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
                excludes = ["./.[^.]*", "./Analysis", "./Production", "./Evaluation", "./Core", "./Preprocessing", "./RunKit", "./soft", "./data", "./tarballs", "*/outputs", "*/mlruns", "__pycache__"]
                exclude_str = " ".join([f"--exclude={ex}" for ex in excludes])
                os.system(f'tar {exclude_str} -czf {tarball_local.path}  .')
            config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
            config.output_files.append("mlruns.tar.gz")
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
        config.custom_content.append(("stream_error", "True"))
        config.custom_content.append(("stream_output", "True"))
        return config

    def create_branch_map(self):
        # Opening file
        print(f"Reading commands from file: {self.input_cmds}")
        self.cmds_list = {}
        with open(self.input_cmds, 'r') as file1:
            for i, line in enumerate(file1):
                self.cmds_list[i] = {"command": line}
        return self.cmds_list

    def output(self):
        # print("HERE","files/mlruns_{}To{}.tar.gz".format(self.branch, int(self.branch) + 1))
        # print("HERE",self.local_target("files/mlruns_{}To{}.tar.gz".format(self.branch, int(self.branch) + 1)))
        if (self.comp_facility == "ETP" and not os.getenv("LAW_JOB_INIT_DIR")):
            # If run on ETP, check if the result .tar is present 
            return self.local_target("files/mlruns_{}To{}.tar.gz".format(self.branch, int(self.branch) + 1))
        else:
            return self.local_target("empty_file_{}.txt".format(self.branch))

    def run(self):
        if not os.path.exists(os.path.abspath(self.working_dir)):
            raise Exception('Working folder {} does not exist'.format(self.working_dir))
        
        command = self.branch_data["command"]
        # match_conf = re.search(r'--config-name\s+(\S+)(\s|$)', command)
        match_inp = re.search(r'input_files=(\S+)(\s|$)', command)
        # Currently hardcoded
        cfg = "configs/train.yaml"
        # cfg = match_conf.group(1)
        input_files_cfg = match_inp.group(1)
        position_from_law_dir = "../"
        full_cfg = position_from_law_dir + self.working_dir + cfg
        
        # Copy in training files
        with initialize(version_base=None, config_path=os.path.dirname(full_cfg)): 
            cfg_data = compose(config_name=os.path.basename(full_cfg), overrides=[f"input_files={input_files_cfg}"])
        
        paths_cfg = cfg_data["input_files"]["cfg"]
        paths_train = cfg_data["input_files"]["train"]
        paths_val = cfg_data["input_files"]["val"]
        
        mass_copy(paths_cfg, os.path.abspath(f"{self.working_dir}/data/cfg.yaml"))
        mass_copy(paths_train, os.path.abspath(f"{self.working_dir}/data/train"), max_workers=64)
        mass_copy(paths_val, os.path.abspath(f"{self.working_dir}/data/val"), max_workers=64)
        
        self.run_command_readable(command, run_location=self.working_dir)
        # self.run_command(command, run_location=self.working_dir)
        if self.comp_facility == "ETP":
            self.run_command(
                "tar -czf ${{LAW_JOB_INIT_DIR}}/mlruns_{}To{}.tar.gz mlruns".format(
                    self.branch, 
                    int(self.branch) + 1
                ), 
                run_location=self.working_dir
            )        
        taskout = self.output()
        taskout.dump('Task ended with no error.')
