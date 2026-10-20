declare module "*.mdx" {
  import type { ComponentType } from "react";
  const C: ComponentType<any>;
  export default C;
}
