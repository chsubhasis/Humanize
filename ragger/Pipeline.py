import logging
from typing import List, Dict, Any
from DomainKnowledge import DomainKnowledgeRAG
import os
from MultiAgent import MultiAgentBRDGenerator


class BRDGenerationPipeline:
    def __init__(
        self,
        document_paths: List[str],
        output_dir: str = 'generated_brds'
    ):
        """
        Complete BRD Generation Pipeline

        Args:
            document_paths (List[str]): Paths to assessment documents
            output_dir (str): Directory to save generated BRDs
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize RAG System
        self.rag_system = DomainKnowledgeRAG()

        # Load and process documents
        self.documents = self.rag_system.load_documents(document_paths)

        # Create vector store
        self.rag_system.create_vector_store(self.documents)

        # Generate synthetic data
        self.synthetic_docs = self.rag_system.generate_synthetic_data(
            self.documents
        )

        # Initialize Multi-Agent System
        self.agent_system = MultiAgentBRDGenerator(self.rag_system)

        self.output_dir = output_dir

    def generate_brds(self):
        """
        Generate BRDs for all input documents

        Returns:
            List of generated BRDs
        """
        generated_brds = []

        for doc in self.documents:
            try:
                # Extract Key Information
                extracted_info = self.agent_system.extract_key_info([doc])

                # Generate BRD
                generated_brd = self.agent_system.generate_brd(extracted_info)

                # Validate BRD
                validation_result = self.agent_system.validate_brd(
                    generated_brd,
                    [doc]
                )

                if validation_result['is_valid']:
                    # Save BRD
                    output_path = os.path.join(
                        self.output_dir,
                        f'BRD_{hash(doc.page_content)}.txt'
                    )

                    with open(output_path, 'w') as f:
                        f.write(generated_brd)

                    generated_brds.append(generated_brd)
                    logging.info(f"Successfully generated BRD: {output_path}")
                else:
                    logging.warning(
                        "BRD validation failed. Skipping document."
                    )

            except Exception as e:
                logging.error(f"Error processing document: {e}")

        return generated_brds


if __name__ == "__main__":
    document_paths = [
    'new_assessment.pdf',
    'new_assessment - Copy.pdf'
    ]
    brd_pipeline = BRDGenerationPipeline(document_paths)
    generated_brds = brd_pipeline.generate_brds()