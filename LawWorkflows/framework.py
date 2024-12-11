# coding: utf-8

import copy
import os
import math

import luigi
import law
from datetime import datetime
from getpass import getuser
from tempfile import mkdtemp
law.contrib.load("htcondor")
law.contrib.load("wlcg")

if os.getenv("LOCAL_TIMESTAMP"):
    startup_time = os.getenv("LOCAL_TIMESTAMP")
else:
    startup_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

def copy_param(ref_param, new_default):
    param = copy.deepcopy(ref_param)
    param._default = new_default
    return param

class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter(
      default="default/{}".format(startup_time),
      description="Versions of runs. Set to a timestamp as default."
    )
    wlcg_path = luigi.Parameter(description="Base-path to remote file location.")
    local_output_path = luigi.Parameter(
        description="Base-path to local file location.",
        default=os.getenv("ANALYSIS_DATA_PATH"),
    )
    is_local_output = luigi.BoolParameter(
        description="Whether to use local storage. False by default."
    )
    try:
        local_user = getuser()
    except:
        pass

    def store_parts(self):
      return (self.__class__.__name__, self.version)

    # def local_path(self, *path):
    #   # ANALYSIS_DATA_PATH is defined in env.sh
    #   parts = (os.getenv("ANALYSIS_DATA_PATH"),) + self.store_parts() + path
    #   return os.path.join(*parts)

    # def local_target(self, *path):
    #   return law.LocalFileTarget(self.local_path(*path))
    
    

    # Path of local targets.
    #   Composed from the analysis path set during the setup.sh
    #   or the local_output_path if is_local_output is set,
    #   the production_tag, the name of the task and an additional path if provided.
    def local_path(self, *path):
        return os.path.join(
            (
                self.local_output_path
                if self.is_local_output
                else os.getenv("ANALYSIS_DATA_PATH")
            ),
            *self.store_parts(),
            *path,
        )

    def temporary_local_path(self, *path):
        if os.environ.get("_CONDOR_JOB_IWD"):
            prefix = os.environ.get("_CONDOR_JOB_IWD") + "/tmp/"
        else:
            prefix = f"/tmp/{self.local_user}"
        temporary_dir = mkdtemp(dir=prefix)
        parts = (temporary_dir,) + (self.__class__.__name__,) + path
        return os.path.join(*parts)

    def local_target(self, path):
        if isinstance(path, (list, tuple)):
            return [law.LocalFileTarget(self.local_path(p)) for p in path]
        return law.LocalFileTarget(self.local_path(path))

    def local_directory_target(self, path):
        if isinstance(path, (list, tuple)):
            return [law.LocalDirectoryTarget(self.local_path(p)) for p in path]
        return law.LocalDirectoryTarget(self.local_path(path))

    def temporarylocal_target(self, *path):
        return law.LocalFileTarget(self.temporary_local_path(*path))

    # Path of remote targets. Composed from the production_tag,
    #   the name of the task and an additional path if provided.
    #   The wlcg_path will be prepended for WLCGFileTargets
    def remote_path(self, *path):
        parts = (self.version,) + (self.__class__.__name__,) + path
        return os.path.join(*parts)

    def remote_target(self, path):
        if self.is_local_output:
            return self.local_target(path)

        if isinstance(path, (list, tuple)):
            return [law.wlcg.WLCGFileTarget(self.remote_path(p)) for p in path]

        return law.wlcg.WLCGFileTarget(self.remote_path(path))

    def remote_directory_target(self, path):
        if self.is_local_output:
            return self.local_directory_target(path)

        if isinstance(path, (list, tuple)):
            return [law.wlcg.WLCGDirectoryTarget(self.remote_path(p)) for p in path]

        return law.wlcg.WLCGDirectoryTarget(self.remote_path(path))


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """
    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime")
    max_memory  = luigi.Parameter(default = '2000', significant = False, description = 'maximum RAM usage')
    batch_name  = luigi.Parameter(default = 'TauML', description = 'HTCondor batch name')
    environment = luigi.ChoiceParameter(default = "", choices = ['', 'cmssw', 'conda', 'cmssw_conda'], var_type = str,
                                        description = "Environment used to run the job")
    requirements = luigi.Parameter(default='', significant=False, description='Requirements for HTCondor nodes')
    max_disk  = luigi.Parameter(default = 'None', significant = False, description = 'maximum scratch space usage')
    num_CPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested CPU.')
    accounting_group   = luigi.Parameter(default = "1", significant = False, description = 'Accounting used for TOpAS.')
    poll_interval = copy_param(law.htcondor.HTCondorWorkflow.poll_interval, 5) # set poll interval to 5 minutes

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(os.getenv("ANALYSIS_PATH"), "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        report_dir = str(self.htcondor_output_directory().path)
        for name in ['error', 'output', 'log']:
            log_dir = os.path.join(report_dir, f'{name}s')
            os.makedirs(log_dir, exist_ok=True)
            config.custom_content.append((name, os.path.join(log_dir, f'{name}.{job_num}.$(ClusterId).$(ProcId).txt')))

        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")
        config.render_variables["environment"] = self.environment
        config.render_variables["LOCAL_TIMESTAMP"] = startup_time
        if 'CONDA_EXE' in os.environ:
            config.render_variables["conda_path"]    = '/'.join(os.environ['CONDA_EXE'].split('/')[:-2])

        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        if len(self.requirements) > 0:
            config.custom_content.append(("requirements", self.requirements))

        config.custom_content.append(('request_memory', f'{self.max_memory}'))
        config.custom_content.append(('request_cpus', self.num_CPUs))
        config.custom_content.append(('JobBatchName', self.batch_name))
        config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
        htcondor_user_proxy = law.wlcg.get_vomsproxy_file()
        config.custom_content.append(("x509userproxy", htcondor_user_proxy))
        config.custom_content.append(('accounting_group', self.accounting_group))

        return config
