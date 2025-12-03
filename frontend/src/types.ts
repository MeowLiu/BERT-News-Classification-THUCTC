export interface Probability {
  name: string;
  value: number;
}

export interface PredictResponse {
  category: string;
  confidence: number;
  probabilities: Probability[];
  inference_time_ms: number;
}

export enum ModelType {
  BERT = 'BERT',
  ROBERTA = 'RoBERTa',
}

export interface DragDropState {
  isDragging: boolean;
  message: string;
}