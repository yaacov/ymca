#!/usr/bin/env python3
"""
Test script for improved question generation quality.

This demonstrates how the improved prompts generate better questions
for retrieval compared to generic questions.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ymca.core.model_handler import ModelHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def analyze_question_quality(questions: list) -> dict:
    """Analyze the quality metrics of generated questions."""
    metrics = {
        'total': len(questions),
        'avg_length': sum(len(q.split()) for q in questions) / len(questions) if questions else 0,
        'question_types': 0,  # Different question types (what, how, when, why, which)
        'too_short': 0,
        'too_long': 0,
        'ideal_length': 0,
        'diverse_starters': set()  # Track unique question starters
    }
    
    # Different question types indicating diversity
    question_starters = ['what', 'how', 'which', 'where', 'when', 'why', 'who']
    
    for q in questions:
        q_lower = q.lower()
        word_count = len(q.split())
        
        # Check for diverse question types
        for starter in question_starters:
            if q_lower.startswith(starter):
                metrics['diverse_starters'].add(starter)
                break
        
        # Length analysis
        if word_count < 5:
            metrics['too_short'] += 1
        elif word_count > 25:
            metrics['too_long'] += 1
        elif 8 <= word_count <= 20:
            metrics['ideal_length'] += 1
    
    metrics['question_types'] = len(metrics['diverse_starters'])
    
    return metrics

def test_question_generation():
    """Test question generation with sample technical documentation."""
    
    print("=" * 80)
    print("Question Generation Quality Test")
    print("=" * 80)
    
    # Initialize model handler
    print("\n1. Initializing model handler...")
    model_handler = ModelHandler(
        model_path="models/ibm-granite_granite-4.0-h-tiny/granite-4.0-h-tiny-q8_0.gguf"
    )
    
    # Sample technical documentation chunks
    test_chunks = [
        {
            "title": "Authentication Configuration",
            "text": """To configure authentication, you need to create a credentials file with your username and password. 
The file should be in JSON format with 'user' and 'password' fields. Place this file in the config directory 
and ensure it has read-only permissions (chmod 400). The application will automatically load credentials on startup.
For certificate-based authentication, provide the certificate path and key file in the configuration."""
        },
        {
            "title": "Troubleshooting Connection Errors",
            "text": """If you encounter connection timeout errors, first verify network connectivity using ping or telnet. 
Check firewall rules to ensure the required ports (8080, 8443) are open. Enable debug logging by setting 
LOG_LEVEL=DEBUG in the environment variables. Review the logs in /var/log/app/ for detailed error messages.
Common causes include incorrect hostnames, certificate validation failures, and proxy configuration issues."""
        },
        {
            "title": "Database Migration Procedure",
            "text": """Before running migrations, backup your database using pg_dump or mysqldump. 
Run 'migrate --dry-run' to preview changes without applying them. Execute migrations with 'migrate up' command.
If migration fails, rollback using 'migrate down N' where N is the number of steps. Always test migrations 
in staging environment first. Monitor migration progress with the --verbose flag."""
        }
    ]
    
    print("\n2. Generating questions for sample documentation chunks:")
    print("-" * 80)
    
    all_questions = []
    
    for i, chunk_info in enumerate(test_chunks, 1):
        print(f"\n📄 Chunk {i}: {chunk_info['title']}")
        print(f"   Text preview: {chunk_info['text'][:100]}...")
        
        # Generate questions
        model_handler.reset_state()
        questions = model_handler.generate_questions(chunk_info['text'], num_questions=3)
        all_questions.extend(questions)
        
        print(f"\n   Generated Questions:")
        for j, q in enumerate(questions, 1):
            word_count = len(q.split())
            print(f"   {j}. {q}")
            print(f"      (length: {word_count} words)")
        
        # Analyze quality
        metrics = analyze_question_quality(questions)
        print(f"\n   Quality Metrics:")
        print(f"   - Avg length: {metrics['avg_length']:.1f} words")
        print(f"   - Question diversity: {metrics['question_types']} different types ({', '.join(sorted(metrics['diverse_starters']))})")
        print(f"   - Ideal length (8-20 words): {metrics['ideal_length']}/{metrics['total']}")
        if metrics['too_short'] > 0:
            print(f"   ⚠️  Too short: {metrics['too_short']}")
        if metrics['too_long'] > 0:
            print(f"   ⚠️  Too long: {metrics['too_long']}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Question Quality Analysis")
    print("=" * 80)
    
    overall_metrics = analyze_question_quality(all_questions)
    
    print(f"\nTotal questions generated: {overall_metrics['total']}")
    print(f"Average length: {overall_metrics['avg_length']:.1f} words")
    print(f"Question diversity: {overall_metrics['question_types']} different types used ({', '.join(sorted(overall_metrics['diverse_starters']))})")
    print(f"Ideal length (8-20 words): {overall_metrics['ideal_length']}/{overall_metrics['total']} ({overall_metrics['ideal_length']/overall_metrics['total']*100:.1f}%)")
    
    if overall_metrics['too_short'] > 0:
        print(f"⚠️  Too short (<5 words): {overall_metrics['too_short']}")
    if overall_metrics['too_long'] > 0:
        print(f"⚠️  Too long (>25 words): {overall_metrics['too_long']}")
    
    # Quality score based on diversity and length
    # Diversity: Using 3+ different question types is good (max 7 types possible)
    diversity_score = min(overall_metrics['question_types'] / 3.0, 1.0) * 50
    length_score = (overall_metrics['ideal_length'] / overall_metrics['total']) * 50
    quality_score = diversity_score + length_score
    
    print(f"\n📊 Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 80:
        print("   ✅ Excellent - Questions are detailed and actionable")
    elif quality_score >= 60:
        print("   ✓ Good - Most questions are well-formed")
    elif quality_score >= 40:
        print("   ⚠️  Fair - Questions need improvement")
    else:
        print("   ❌ Poor - Questions are too generic")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
    
    # Show example comparison
    print("\n💡 Question Quality Comparison:")
    print("\nBAD (not diverse, repeating same pattern):")
    print("  - What is cold migration?")
    print("  - What is warm migration?")
    print("  - What is live migration?")
    print("  ← All ask the same thing about different items!")
    print("\nGOOD (diverse aspects of the same content):")
    if all_questions:
        for q in all_questions[:3]:
            print(f"  - {q}")
        print("  ← Each covers a different aspect!")

if __name__ == "__main__":
    test_question_generation()

