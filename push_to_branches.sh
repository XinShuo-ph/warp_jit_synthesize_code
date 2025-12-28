#!/bin/bash
cd "$(dirname "$0")"

# Save the instructions file content
cp instructions_wrapup.md /tmp/instructions_wrapup.md

# List of branch suffixes
branches="0fbe 12c4 25e7 3576 3a5b 3f34 4b76 5d09 7288 82cf 8631 9177 a4fd aa30 d623 ff72"

for suffix in $branches; do
  branch="cursor/following-instructions-md-$suffix"
  echo "=== Processing $branch ==="
  
  # Checkout the branch
  git checkout -B $branch origin/$branch
  if [ $? -ne 0 ]; then
    echo "Failed to checkout $branch, skipping..."
    continue
  fi
  
  # Find where jit folder is and copy the instructions file
  if [ -d "jit" ]; then
    cp /tmp/instructions_wrapup.md jit/instructions_wrapup.md
  else
    cp /tmp/instructions_wrapup.md instructions_wrapup.md
  fi
  
  # Add and commit
  git add -A
  git commit -m "Add wrapup instructions for branch consolidation" --allow-empty
  
  # Push
  git push origin $branch
  
  echo ""
done

# Return to main
git checkout main
echo "Done! Pushed to all 16 branches."

