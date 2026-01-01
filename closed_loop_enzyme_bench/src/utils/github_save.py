"""GitHub 결과 저장 헬퍼 (Colab용)"""
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def save_results_to_github(
    experiment_name: str,
    github_token: Optional[str] = None,
    repo_url: Optional[str] = None,
    branch: str = "main"
) -> bool:
    """
    실험 결과(CSV, PNG)를 GitHub에 자동 커밋 & 푸시
    
    Args:
        experiment_name: 실험 이름 (예: "exp02_single_shot")
        github_token: GitHub Personal Access Token (None이면 Colab Secrets에서 가져옴)
        repo_url: 저장소 URL (None이면 현재 git remote 사용)
        branch: 브랜치 이름 (기본: "main")
    
    Returns:
        성공 여부
    """
    try:
        # 1. Git 설정 확인
        if not Path('.git').exists():
            print("⚠ Git repository not found. Initialize first:")
            print("  !git clone https://github.com/YOUR_USERNAME/bioinformatics.git")
            return False
        
        # 2. GitHub Token 가져오기 (Colab Secrets 우선)
        if github_token is None:
            try:
                from google.colab import userdata
                github_token = userdata.get('GITHUB_TOKEN')
            except ImportError:
                # Not in Colab, try environment variable
                github_token = os.environ.get('GITHUB_TOKEN')
            except Exception:
                pass
            
            if not github_token:
                print("⚠ GitHub token not found. Set it in Colab Secrets:")
                print("  - Click 🔑 icon → Add secret")
                print("  - Name: GITHUB_TOKEN")
                print("  - Value: Your GitHub Personal Access Token")
                print("  Or set environment variable: GITHUB_TOKEN")
                return False
        
        # 3. 원격 저장소 설정
        if repo_url:
            # URL에서 username 추출
            if 'github.com' in repo_url:
                repo_url = repo_url.replace('https://', f'https://{github_token}@')
            subprocess.run(['git', 'remote', 'set-url', 'origin', repo_url], 
                         check=False, capture_output=True)
        
        # 4. 결과 파일 확인
        csv_files = list(Path("results/tables").glob("*.csv"))
        png_files = list(Path("results/figures").glob("*.png"))
        
        if not csv_files and not png_files:
            print("⚠ No results to commit (no CSV or PNG files found)")
            print(f"  Checked: results/tables/*.csv, results/figures/*.png")
            return False
        
        # 5. Git add (결과 파일만)
        added = False
        for f in csv_files + png_files:
            result = subprocess.run(['git', 'add', str(f)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                added = True
                print(f"  ✓ Added: {f.name}")
        
        if not added:
            print("⚠ No new files to commit")
            return False
        
        # 6. 커밋
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        commit_msg = f"Results: {experiment_name} - {timestamp}"
        
        result = subprocess.run(['git', 'commit', '-m', commit_msg],
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            if "nothing to commit" in result.stdout.lower() or "nothing to commit" in result.stderr.lower():
                print("ℹ No changes to commit")
                return True
            print(f"⚠ Commit failed: {result.stderr}")
            return False
        
        print(f"  ✓ Committed: {commit_msg}")
        
        # 7. 푸시
        result = subprocess.run(['git', 'push', 'origin', branch],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Results saved to GitHub: {commit_msg}")
            return True
        else:
            print(f"⚠ Push failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error saving to GitHub: {e}")
        import traceback
        traceback.print_exc()
        return False
