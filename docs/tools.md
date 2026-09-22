# Tools

The orchestration API keeps tools internal. Tools are registered with schemas, checked by the permission gate, validated before execution, sanitized after execution, and tracked by the loop guard and provenance layer. Do not register tools that make destructive external changes without a confirmation requirement.
