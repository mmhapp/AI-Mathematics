# AI Mathematics

This umbrella repository links three projects on [Backpropagation](https://github.com/mmhapp/Backpropagation), [Optimisers](https://github.com/mmhapp/Optimisers), and [Diffusion](https://github.com/mmhapp/Diffusion). Projects rebuild the core ideas of machine learning (ML) from scratch and open up the mathematics behind them. These building blocks appear everywhere in modern artificial intelligence (AI) – optimisers train large language models, backpropagation is how every neural network learns, and diffusion is used in generative models.

---

## Repositories

The core ideas behind [Backpropagation](https://github.com/mmhapp/Backpropagation) are gradient of loss function and chain rule.

The gradient of the loss is propagated backwards through the network with the chain rule as $\delta^{(l)} = \bigl(\delta^{(l+1)}(W^{(l+1)})^\top\bigr)\odot\sigma'(z^{(l)})$.

The parameter gradients follow directly as $\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^\top\delta^{(l)},\qquad\frac{\partial L}{\partial b^{(l)}} = \sum_i\delta^{(l)}_i$.

--

The core ideas behind [Optimisers](https://github.com/mmhapp/Optimisers) are Gradient Descent, Momentum, and Adam.

Gradient descent as

$\theta \leftarrow \theta - \eta\,\nabla_\theta L$.

Momentum as

$u_t = \beta u_{t-1} + g_t,\qquad
\theta \leftarrow \theta - \eta\,u_t$.

Adam as

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, &\quad \hat m_t &= \frac{m_t}{1-\beta_1^t},\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, &\quad \hat v_t &= \frac{v_t}{1-\beta_2^t},\\
\theta &\leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

--

The core idea behind [Diffusion](https://github.com/mmhapp/Diffusion) is the forward noising process that can be written in closed form as

$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I)$.
