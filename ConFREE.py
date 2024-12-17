import torch
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

class ParameterAggregator:
    def __init__(self, global_model):
        self.global_model = global_model
        self.previous_global_model = None
        self.rng = np.random.default_rng()

    def aggregate_parameters_wise(self, global_model,uploaded_models,alpha):
        """
        The global parameters for the round are obtained by using the global parameters of the previous round plus 
        the parameters of each client for that round minus the global parameters of the previous round multiplied 
        by the weights of each client sample
        """
        assert len(uploaded_models) > 0, "Ensure at least one client model has been uploaded"
        self.global_model = global_model
        # Using the global model from the previous round as a baseline
        self.previous_global_model = deepcopy(self.global_model)
        self.global_model = deepcopy(uploaded_models[0])
        
        
        for param in self.global_model.parameters():
            param.data.zero_()  # 所有参数清零

        # Calculate updated values for all clients
        client_updates = self.calculate_client_updates(uploaded_models)

        # Handling Client Update Conflicts
        reduced_client_updates = self.resolve_conflicts(client_updates, alpha)
       
        # Update mean
        avg_updates = self.aggregate_parameters(reduced_client_updates)

        # add the global parameters from the previous round
        for global_param, previous_param in zip(self.global_model.parameters(), self.previous_global_model.parameters()):
            global_param.data += previous_param.data.clone()
        
        self.previous_global_model = None

        return self.global_model

    def aggregate_parameters(self, client_updates):
        
        num_clients = len(client_updates)
        total_updates = [torch.zeros_like(param.data) for param in self.global_model.parameters()]
        # Iterate through each client's updates
        for client_update in client_updates:
            for total_update, update_param in zip(total_updates, client_update.parameters()):
                total_update += update_param.data.clone()
        for server_param, total_update in zip(self.global_model.parameters(), total_updates):
            server_param.data += total_update / num_clients

        avg_updates = [total_update / num_clients for total_update in total_updates]
        
        return avg_updates

    def calculate_client_updates(self, client_models):
        client_updates = []
        for client_model in client_models:
            update_model = deepcopy(self.previous_global_model)
            for update_param, server_param, client_param in zip(update_model.parameters(), self.previous_global_model.parameters(), client_model.parameters()):
                update_param.data = client_param.data.clone() - server_param.data.clone()
            client_updates.append(update_model)
        return client_updates

    def resolve_conflicts(self, client_updates, alpha):
        num_clients = len(client_updates)
        client_updates_params = [list(client_update.parameters()) for client_update in client_updates]
        num_layers = len(client_updates_params[0])
        for layer in range(num_layers):
            clients = []
            for i in range(num_clients):
                clients.append(client_updates_params[i][layer].data.view(-1))
            clients = torch.stack(clients, dim=1)  # [num_params, num_clients]
            new_grad = self.cal_optimum(clients,alpha)
            for i in range(num_clients):
                client_updates_params[i][layer].data.copy_(new_grad.view(client_updates_params[i][layer].data.size()))
        return client_updates

    def cal_optimum(self, clients, alpha, rescale=1):
        GG = clients.t().mm(clients).cpu()  # [num_tasks, num_tasks]
        # rng = np.random.default_rng()
        c0_orial = self.cal_guid(clients, self.rng)
        c0_norm = c0_orial.norm().cpu()

        n_tasks = clients.shape[1]
        x_start = np.ones(n_tasks) / n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
        A = GG.numpy()
        c0 = torch.tensor(c0_orial.cpu().numpy()).view(-1, 1).to(clients.device)
        GT_c0 = clients.t().mm(c0).cpu().numpy()
        c = (alpha * c0_norm + 1e-8).item()

        def objfn(x):
            term1 = np.dot(x, GT_c0)
            term2 = c * np.sqrt(x.dot(A).dot(x) + 1e-8)
            return term1 + term2

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(clients.device)
        gw = (clients * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = c0_orial + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    def cal_guid(self, clients, rng):
        grad_vec = clients.t()  # [num_tasks, num_params]
        shuffled_task_indices = np.zeros((clients.shape[1], clients.shape[1] - 1), dtype=int)
        for i in range(clients.shape[1]):
            task_indices = np.arange(clients.shape[1])
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T
        normalized_grad_vec = grad_vec 

        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[task_indices]  # [num_tasks, num_params]
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)  # [num_tasks, 1]
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad  # [num_tasks, num_params]  
        g = modified_grad_vec.mean(dim=0)  # [num_params]

        return g