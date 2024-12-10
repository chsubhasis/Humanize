from DomainKnowledge import DomainKnowledgeRAG

class MultiAgentBRDGenerator:
    def __init__(self, rag_system: DomainKnowledgeRAG):
        """
        Multi-agent system for BRD generation

        Args:
            rag_system (DomainKnowledgeRAG): RAG system for knowledge injection
        """
        self.rag_system = rag_system

    def extract_key_info(self, documents):
        """
        Agent for extracting key information from assessment reports.
        You can customize and add logics to parse different assessment files.

        Args:
            documents (List): Assessment documents

        Returns:
            Dict of extracted key information
        """
        extraction_prompt = """
        Extract the following key information from the document:
        1. Business Objectives
        2. Functional Requirements
        3. Non-Functional Requirements
        4. Constraints and Limitations

        Document Context: {document}

        Extracted Information:
        """

        extracted_info = self.rag_system.llm(
            extraction_prompt.format(document=documents[0].page_content)
        )

        return extracted_info

    def generate_brd(self, extracted_info):
        """
        Agent for generating Business Requirement Document

        Args:
            extracted_info (Dict): Key information from assessments

        Returns:
            str: Generated BRD
        """
        brd_generation_prompt = """
        Using the following extracted information, create a comprehensive
        Business Requirement Document (BRD) for SAP/SFDC domain:

        {extracted_info}

        BRD Structure:
        1. Executive Summary
        2. Detailed Business Requirements
        3. Functional Specifications
        4. Non-Functional Requirements
        5. Constraints and Assumptions
        6. Appendices

        Generated BRD:
        """

        generated_brd = self.rag_system.llm(
            brd_generation_prompt.format(extracted_info=extracted_info)
        )

        return generated_brd