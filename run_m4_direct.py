#!/usr/bin/env python
"""Run Module 4 main() directly in real-data mode."""
import sys
import os

print("=" * 70)
print("Module 4 — Real Data Fairness Audit (Direct Execution)")
print("=" * 70)
print()

# Import and run main()
try:
    from src.fairness_audit import main
    print("✓ Imported main() successfully\n")
    
    print("Executing main() with real data...\n")
    main()
    
    print("\n" + "=" * 70)
    print("Module 4 Completed Successfully!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error in main(): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
