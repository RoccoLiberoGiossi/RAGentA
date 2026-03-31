import os
import torch
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

class BaseLLMAgent(ABC):
    """Base class for all LLM agents."""
    
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def get_log_probs(self, prompt: str, target_tokens: List[str] = ["Yes", "No"]) -> Dict[str, float]:
        """Calculate log probabilities for specific tokens."""
        pass

    def batch_process(self, prompts: List[str], generate: bool = True, max_new_tokens: int = 256) -> List[Any]:
        """Process a batch of prompts."""
        results = []
        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))
        return results

class HuggingFaceAgent(BaseLLMAgent):
    """LLM agent that uses local HuggingFace models."""
    
    def __init__(self, model_name: str, device: str = "cuda", precision: str = "bfloat16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Initializing HuggingFaceAgent with model: {model_name}")
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        if token:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        if precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("Using bfloat16 precision")
        elif precision == "float16":
            torch_dtype = torch.float16
            print("Using float16 precision")
        else:
            torch_dtype = torch.float32
            print("Using float32 precision")
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=token if token else None,
        )
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Model loaded on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        # Format as proper chat messages using the tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]
        # Apply the model's chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize the formatted prompt
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
        
        # Generate the response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = model_inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True).strip()
        
        if not generated_text:
            return "I don't have enough information to provide a specific answer."
            
        return generated_text

    def get_log_probs(self, prompt: str, target_tokens: List[str] = ["Yes", "No"]) -> Dict[str, float]:
        """
        Calculate log probabilities for specific tokens.

        Args:
            prompt: The input prompt
            target_tokens: List of tokens to get probabilities for

        Returns:
            Dictionary mapping tokens to their log probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        target_ids = []
        for token in target_tokens:
            token_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            target_ids.append(token_ids[0] if token_ids else self.tokenizer.unk_token_id)
            
        log_probs = torch.log_softmax(logits, dim=0)
        return {token: log_probs[tid].item() for token, tid in zip(target_tokens, target_ids)}

class OpenAIAgent(BaseLLMAgent):
    """LLM agent that uses OpenAI API or OpenAI-compatible APIs."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_base: Optional[str] = None):
        from openai import OpenAI
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        print(f"OpenAIAgent initialized with model: {model_name}")

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            generated_text = response.choices[0].message.content.strip()
            
            if not generated_text:
                return "I don't have enough information to provide a specific answer."
                
            return generated_text
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error: {e}"

    def get_log_probs(self, prompt: str, target_tokens: List[str] = ["Yes", "No"]) -> Dict[str, float]:
        try:
            # Note: OpenAI's logprobs behavior is slightly different.
            # We'll use a single-token completion and check logprobs of the first token.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
            )
            
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            result = {token: -100.0 for token in target_tokens} # Default low prob
            
            for logprob_item in top_logprobs:
                for token in target_tokens:
                    if logprob_item.token.strip().lower() == token.lower():
                        result[token] = logprob_item.logprob
            return result
        except Exception as e:
            print(f"OpenAI logprobs error: {e}")
            return {token: -100.0 for token in target_tokens}

class VLLMAgent(BaseLLMAgent):
    """LLM agent that uses vLLM (local or via OpenAI-compatible API)."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_base: Optional[str] = None, is_local: bool = True):
        self.model_name = model_name
        self.is_local = is_local
        
        if is_local:
            from vllm import LLM
            print(f"Initializing local vLLM with model: {model_name}")
            self.llm = LLM(model=model_name, trust_remote_code=True)
        else:
            from openai import OpenAI
            self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
            self.api_base = api_base or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            print(f"VLLMAgent initialized via API: {self.model_name}")

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        if self.is_local:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            outputs = self.llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                generated_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"vLLM API error: {e}")
                return f"Error: {e}"
        
        if not generated_text:
            return "I don't have enough information to provide a specific answer."
            
        return generated_text

    def get_log_probs(self, prompt: str, target_tokens: List[str] = ["Yes", "No"]) -> Dict[str, float]:
        if self.is_local:
            from vllm import SamplingParams
            # Request logprobs for the generated token
            sampling_params = SamplingParams(
                temperature=0.0, 
                max_tokens=1, 
                logprobs=20 # Get top 20 logprobs
            )
            outputs = self.llm.generate([prompt], sampling_params)
            
            # Extract logprobs from the first (and only) generated token
            logprobs_list = outputs[0].outputs[0].logprobs
            if not logprobs_list:
                return {token: -100.0 for token in target_tokens}
                
            token_logprobs = logprobs_list[0]
            result = {token: -100.0 for token in target_tokens}
            
            # vLLM returns a dict of {token_id: Logprob} or similar depending on version
            # We need to iterate and match tokens
            for tid, logprob_obj in token_logprobs.items():
                # In newer vLLM, logprob_obj might be a Logprob object with a 'decoded_token'
                token_text = getattr(logprob_obj, 'decoded_token', str(tid)).strip().lower()
                for target in target_tokens:
                    if token_text == target.lower():
                        result[target] = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else float(logprob_obj)
            return result
        else:
            # Use same logic as OpenAIAgent for API
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=5,
                )
                top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                result = {token: -100.0 for token in target_tokens}
                for logprob_item in top_logprobs:
                    for token in target_tokens:
                        if logprob_item.token.strip().lower() == token.lower():
                            result[token] = logprob_item.logprob
                return result
            except Exception as e:
                print(f"vLLM API logprobs error: {e}")
                return {token: -100.0 for token in target_tokens}

def get_llm_agent(interface_type: str, model_name: str, **kwargs) -> BaseLLMAgent:
    """Factory function to get the appropriate LLM agent."""
    if interface_type.lower() == "huggingface":
        return HuggingFaceAgent(model_name, **kwargs)
    elif interface_type.lower() == "openai":
        return OpenAIAgent(model_name, **kwargs)
    elif interface_type.lower() == "vllm":
        return VLLMAgent(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")
