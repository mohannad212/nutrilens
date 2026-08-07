// Mirrors backend/app/schemas/recognition.py field-for-field (03-api-specification.md §4).
// `portion` is intentionally absent - that field lands in M4 (05-portion-estimation.md).

export interface Prediction {
  rank: number;
  food101_class: string;
  display_name: string;
  confidence: number;
}

export interface Alternative {
  food101_class: string;
  display_name: string;
  confidence: number;
}

export interface RecognitionResponse {
  request_id: string;
  items: Prediction[];
  low_confidence: boolean;
  alternatives: Alternative[];
}
