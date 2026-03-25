import os
import shutil
from pathlib import Path
import pdfplumber
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import numpy as np

class AcademicPaperOrganizer:
    """academic paper classification"""
    
    def __init__(self, source_dir, output_dir=None):
        self.source_dir = Path(source_dir)
        if output_dir is None:
            self.output_dir = self.source_dir / "classified_papers"
        else:
            self.output_dir = Path(output_dir)
        
        self.papers = []
        self.paper_features = []
        
        if not self.source_dir.exists():
            raise ValueError(f"source directory does not exist: {self.source_dir}")
    
    def extract_paper_metadata(self, pdf_path):
        """extract paper metadata"""
        metadata = {
            'path': pdf_path,
            'filename': pdf_path.name,
            'text': '',
            'first_page': '',
            'abstract': '',
            'keywords': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ''
                for i, page in enumerate(pdf.pages[:5]):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n{text}"
                
                if not full_text.strip():
                    print(f"warning: {pdf_path.name} cannot extract text content")
                    return metadata
                
                metadata['text'] = full_text
                
                metadata['abstract'] = self.extract_abstract(full_text)
                
                metadata['keywords'] = self.extract_keywords(full_text)
                
        except Exception as e:
            print(f"error processing {pdf_path.name}: {e}")
            
        return metadata
    
    def extract_abstract(self, text):
        """extract abstract"""
        patterns = [
            r'Abstract[:\s]+(.*?)(?=\n\n|\n[A-Z]|\n\n[A-Z])',
            r'ABSTRACT[:\s]+(.*?)(?=\n\n|\n[A-Z]|\n\n[A-Z])',
            r'摘要[:\s]+(.*?)(?=\n\n)',
            r'\[Abstract\][:\s]+(.*?)(?=\n\n)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) > 100:
                    return abstract[:2000]
        
        return text[:500]
    
    def extract_keywords(self, text):
        """extract keywords"""
        kw_patterns = [
            r'Keywords?[:\s]+(.*?)(?=\n\n|\n[A-Z])',
            r'KEYWORDS[:\s]+(.*?)(?=\n\n|\n[A-Z])',
            r'关键词[:\s]+(.*?)(?=\n\n)'
        ]
        
        for pattern in kw_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                kw_text = match.group(1)
                keywords = re.split(r'[,;]\s*', kw_text)
                keywords = [k.strip().lower() for k in keywords if len(k.strip()) > 2]
                return keywords[:10]
        
        return self.extract_important_phrases(text)
    
    def extract_important_phrases(self, text, top_n=8):
        """extract important phrases as keywords"""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=top_n,
                min_df=1
            )
            
            X = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            word_counts = X.toarray()[0]
            word_freq = list(zip(feature_names, word_counts))
            word_freq.sort(key=lambda x: x[1], reverse=True)
            
            return [word for word, count in word_freq[:top_n] if len(word) > 2]
        except:
            return []
    
    def compute_similarity_matrix(self):
        """compute paper similarity matrix"""
        texts = []
        valid_indices = []
        
        for i, paper in enumerate(self.papers):
            combined_text = f"{paper['abstract']} "
            if paper['keywords']:
                combined_text += f"{' '.join(paper['keywords'])} "
            combined_text += paper['text'][:5000]  
            
            if combined_text.strip():
                texts.append(combined_text)
                valid_indices.append(i)
            else:
                print(f"warning: {paper['filename']} has no valid text content")
        
        if len(texts) < 2:
            print("valid text insufficient, cannot compute similarity")
            return None, []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            return similarity_matrix, valid_indices
            
        except Exception as e:
            print(f"error computing similarity matrix: {e}")
            return None, []
    
    def cluster_papers(self, similarity_matrix, threshold=0.2):
        """cluster papers based on similarity matrix"""
        try:
            distance_matrix = 1 - similarity_matrix
            
            distance_matrix = np.maximum(distance_matrix, 0)
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            return labels
            
        except Exception as e:
            print(f"error clustering papers: {e}")
            return None
    
    def generate_cluster_names(self, cluster_indices):
        """generate meaningful names for each cluster"""
        cluster_names = {}
        
        for cluster_id, indices in cluster_indices.items():
            all_keywords = []
            all_abstracts = []
            
            for idx in indices:
                paper = self.papers[idx]
                if paper['keywords']:
                    all_keywords.extend(paper['keywords'])
                if paper['abstract']:
                    all_abstracts.append(paper['abstract'])
            
            if all_keywords:
                keyword_counter = Counter(all_keywords)
                top_keywords = [kw.replace(' ', '_') for kw, _ in keyword_counter.most_common(3)]
                cluster_name = '_'.join(top_keywords)
            else:
                cluster_name = self.extract_cluster_name_from_texts(all_abstracts)
            
            if len(cluster_name) > 50:
                cluster_name = cluster_name[:50]
            
            cluster_names[cluster_id] = cluster_name or f"topic_{cluster_id}"
        
        return cluster_names
    
    def extract_cluster_name_from_texts(self, texts):
        """extract cluster name from text collection"""
        if not texts:
            return "unknown_topic"
        
        combined_text = ' '.join(texts)
        keywords = self.extract_important_phrases(combined_text, top_n=3)
        return '_'.join(keywords) if keywords else "unknown_topic"
    
    def organize_papers(self):
        """main function: organize all papers"""
        print("Step 1: extract paper information...")
        
        pdf_files = list(self.source_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"no PDF files found in {self.source_dir}")
            return
        
        print(f"found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            print(f"processing: {pdf_path.name}")
            metadata = self.extract_paper_metadata(pdf_path)
            if metadata['text']:  
                self.papers.append(metadata)
            else:
                print(f"skip: cannot extract text content")
        
        print(f"successfully processed {len(self.papers)} papers")
        
        if len(self.papers) < 2:
            print("paper number insufficient (less than 2), cannot classify")
            return
        
        print("Step 2: compute similarity matrix...")
        similarity_matrix, valid_indices = self.compute_similarity_matrix()
        
        if similarity_matrix is None or len(valid_indices) < 2:
            print("cannot compute valid similarity matrix")
            return
        
        print("Step 3: cluster analysis...")
        labels = self.cluster_papers(similarity_matrix, threshold=0.9)
        
        if labels is None:
            print("cluster analysis failed")
            return
        
        clusters = {}
        for i, label in enumerate(labels):
            original_idx = valid_indices[i]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(original_idx)
        
        print(f"Step 4: generate {len(clusters)} categories...")
        cluster_names = self.generate_cluster_names(clusters)
        
        print("Step 5: create folders and copy files...")
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"output directory: {self.output_dir}")
        except Exception as e:
            print(f"error creating output directory: {e}")
            return
        
        unclassified_dir = self.output_dir / "unclassified"
        
        for cluster_id, indices in clusters.items():
            if len(indices) == 1:
                target_dir = unclassified_dir
            else:
                folder_name = cluster_names[cluster_id]
                target_dir = self.output_dir / folder_name
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for idx in indices:
                paper = self.papers[idx]
                source_path = paper['path']
                target_path = target_dir / source_path.name
                
                if not target_path.exists():
                    try:
                        shutil.copy2(source_path, target_path)
                    except Exception as e:
                        print(f"error copying file: {source_path.name}: {e}")
            
            print(f"  - {target_dir.name}: {len(indices)} papers")
        
        print("\nclassification completed!")
        self.generate_summary_report(clusters, cluster_names, unclassified_dir)
    
    def generate_summary_report(self, clusters, cluster_names, unclassified_dir):
        """generate classification report"""
        report_path = self.output_dir / "classification_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("academic paper classification report\n")
                f.write("="*60 + "\n\n")
                f.write(f"total paper number: {len(self.papers)}\n")
                f.write(f"classification number: {len(clusters)}\n\n")
                
                for cluster_id, indices in clusters.items():
                    if cluster_id in cluster_names:
                        folder_name = cluster_names[cluster_id]
                    else:
                        folder_name = f"cluster_{cluster_id}"
                    
                    if len(indices) == 1:
                        f.write(f"category: unclassified (original plan: {folder_name})\n")
                    else:
                        f.write(f"category: {folder_name}\n")
                    
                    f.write(f"paper number: {len(indices)}\n")
                    f.write("contains papers:\n")
                    for idx in indices:
                        f.write(f"  - {self.papers[idx]['filename']}\n")
                    f.write("\n" + "-"*50 + "\n\n")
            
            print(f"report saved to: {report_path}")
            
        except Exception as e:
            print(f"error generating report: {e}")

# example
if __name__ == "__main__":
    try:
        # please modify to your actual path
        organizer = AcademicPaperOrganizer(
            source_dir=r"C:\\Users\\小耳東同学\\Desktop\\test1\\articles",  # modify to your actual PDF folder path
            output_dir=r"C:\\Users\\小耳東同学\\Desktop\\test1\\class"  # modify to your actual output path
        )
        organizer.organize_papers()
    except Exception as e:
        print(f"error running program: {e}")