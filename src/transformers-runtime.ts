import type { ModelOption } from './model-catalog.js';
import type { ChatMessage } from './session.js';

/** Host-owned worker adapter; this package never imports a heavy ONNX dependency. */
export type TransformersEngine = {
  generate(
    messages: ChatMessage[],
    options: {
      maxOutputTokens: number;
      signal: AbortSignal;
      onUpdate: (text: string) => void;
    },
  ): Promise<{ reply: string; usage: unknown }>;
  countTokens(messages: ChatMessage[]): Promise<number>;
  cancel(): void;
  reset(): Promise<void>;
  dispose(): Promise<void>;
};
export type TransformersRuntime = {
  createEngine(
    model: ModelOption,
    options: {
      signal: AbortSignal;
      onProgress: (progress: Record<string, unknown>) => void;
    },
  ): Promise<TransformersEngine>;
};

/** Which Transformers.js class and files a catalog ONNX model may load. */
export type OnnxTextLoadPlan = {
  modelClass: 'AutoModelForCausalLM' | 'Ministral3ForCausalLM';
  dtype?: string | Record<string, string>;
  /** Omitted for the Ministral text graph so the repo config selects chunk counts per file. */
  useExternalDataFormat?: number;
};

/**
 * LFM keeps the single `model_*.onnx` decoder. Ministral 3 Instruct ONNX is a separate
 * text stack: q4 merged decoder plus fp16 embeddings, and never the vision encoder.
 */
export function onnxTextLoadPlan(
  model: Pick<ModelOption, 'transformersGraph' | 'precision' | 'externalDataFiles'>,
): OnnxTextLoadPlan {
  if (model.transformersGraph === 'ministral3-text') {
    return {
      modelClass: 'Ministral3ForCausalLM',
      dtype: { decoder_model_merged: 'q4', embed_tokens: 'fp16' },
    };
  }
  return {
    modelClass: 'AutoModelForCausalLM',
    dtype: model.precision,
    useExternalDataFormat: model.externalDataFiles,
  };
}
