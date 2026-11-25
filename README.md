# AI Mathematics

This umbrella repository links three projects on [Backpropagation](https://github.com/mmhapp/Backpropagation), [Diffusion](https://github.com/mmhapp/Diffusion), and [Optimisers](https://github.com/mmhapp/Optimisers). Projects rebuild the core ideas of machine learning (ML) from scratch and open up the mathematics behind them. These building blocks appear everywhere in modern artificial intelligence (AI) – optimisers train large language models, backpropagation is how every neural network learns, and diffusion is used in generative models.

---

## Repositories

The core ideas behind [Backpropagation](https://github.com/mmhapp/Backpropagation) are gradient of loss function and chain rule.

The gradient of the loss is propagated backwards through the network with the chain rule as

$\delta^{(l)} = \bigl(\delta^{(l+1)}(W^{(l+1)})^\top\bigr)\odot\sigma'(z^{(l)})$.

The parameter gradients follow directly as

$\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^\top\delta^{(l)},\qquad\frac{\partial L}{\partial b^{(l)}} = \sum_i\delta^{(l)}_i$.

The core idea behind [Diffusion](https://github.com/mmhapp/Diffusion) is the forward noising process, in closed form as

$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I)$.

The core ideas behind [Optimisers](https://github.com/mmhapp/Optimisers) are Gradient Descent, Momentum, and Adam.

Gradient descent as

$\theta \leftarrow \theta - \eta\,\nabla_\theta L$.

Momentum as

$\theta \leftarrow \theta - \eta\,u_t$.

Adam as

$\theta\leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$.
