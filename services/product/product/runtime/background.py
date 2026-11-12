"""Reuse LaMa on its own host under the existing shared GPU file lease."""
from ..parallel_store import BudgetExceeded
from ..recovery.lama_adapter import LamaClient
from ..recovery.product_pipeline import InpaintService
from ..resource_lease import lease


class HostInpaintService:
    def __init__(self,config,cache,budget):
        self.lock=config['local_gpu_lease']
        if set(self.lock)!={'root','name'}:
            raise ValueError('local GPU lease requires its existing root and resource name')
        def dispatch(identity):
            if budget and not budget.reserve('model_calls',{'tool':'big-lama','request_id':identity}):
                raise BudgetExceeded('identical LaMa dispatch already reserved')
        self.client=LamaClient(config['url'],config['model_binding'],cache,on_dispatch=dispatch)
        self.receipt=None

    def __call__(self,source,mask):
        with lease(self.lock['name'],root=self.lock['root'],capacity=1):
            result=self.client(source,mask)
        self.receipt={'request':self.client.receipt,'gpu_lease':{'status':'ACQUIRED',**self.lock}}
        return result


def inpaint_service(config,cache,budget):
    if 'local_gpu_lease' in config:
        return HostInpaintService(config,cache,budget)
    return InpaintService(config,cache,budget)
