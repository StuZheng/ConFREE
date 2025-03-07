# ConFREE: Conflict-free Client Update Aggregation for Personalized Federated Learning

This is the implementation of our paper: ConFREE: Conflict-free Client Update Aggregation for Personalized Federated Learning (AAAI 2025 Oral). ConFREE is the first method to address client update conflicts in PFL. By optimizing global model aggregation, it provides each client with more effective and comprehensive global information.


ConFREE projects conflicting updates onto a normal plane, creates a conflict-free guiding vector $\Delta \bar{\theta}_{\perp}$. The optimal $\Delta \theta^*$ is then found within a ball centered around $\Delta \bar{\theta}_{\perp}$, which maximizes the local improvement of the worst performing client in the neighborhood. This ensures the updated global model is closer to the global optimum $\theta^*_i_j$ and balances the update across clients.



- [Poster](./ConFREE_Poster.pdf)
- [Slides](./Hao%20Zheng_ConFREE_AAAI2025_Oral.pdf)


## 📝 Citation

If you find ConFREE useful for your research, please consider citing our paper:

```


```

## 📦 Algorithms
ConFREE projects conflicting updates onto a normal plane, creates a conflict-free guiding vector $\Delta \bar{\theta}_{\perp}$. The optimal $\Delta \theta^*$ is then found within a ball centered around $\Delta \bar{\theta}_{\perp}$, which maximizes the local improvement of the worst performing client in the neighborhood. This ensures the updated global model is closer to the global optimum $\theta^*_i_j$ and balances the update across clients.



## 📄 README
This README provides guidance on how to set up, run, and extend the code associated with our method, which is designed to be integrated within the PFLlib framework. Our approach is orthogonal to the existing methods in PFLlib, making it possible to combine them for complementary performance gains. The execution environment and dependencies are identical to those of PFLlib to ensure seamless integration.
https://github.com/TsingZ0/PFLlib
