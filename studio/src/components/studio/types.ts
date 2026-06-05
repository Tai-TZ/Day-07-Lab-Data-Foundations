export type Phase = "idle" | "indexing" | "embedding" | "retrieving" | "generating" | "done";

export type NodeId =
  | "pdf"
  | "loader"
  | "chunks"
  | "embedIdx"
  | "vdb"
  | "user"
  | "query"
  | "embedRet"
  | "qvec"
  | "topk"
  | "llm"
  | "answer";
