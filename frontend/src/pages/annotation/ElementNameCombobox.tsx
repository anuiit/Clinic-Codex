import type { DetectedElement } from "../../types";
import { appText } from "../../i18n/text";
import annotationStyles from "./AnnotationChrome.module.css";
import { ClassNameCombobox } from "../../components/ui/ClassNameCombobox";

interface ElementNameComboboxProps {
  value: string;
  classNames: string[];
  customClassNames: string[];
  topK: DetectedElement["top_k"];
  autoFocusToken: number;
  labels: typeof appText.annotation;
  index: number;
  onCommit: (name: string) => void;
}
export function ElementNameCombobox({
  value,
  classNames,
  customClassNames,
  topK,
  autoFocusToken,
  labels,
  index,
  onCommit,
}: ElementNameComboboxProps) {
  return (
    <ClassNameCombobox
      value={value}
      classNames={classNames}
      customClassNames={customClassNames}
      topK={topK}
      autoFocusToken={autoFocusToken}
      labels={labels}
      index={index}
      onCommit={onCommit}
      className={`${annotationStyles.owner} annotation-name-combobox`}
      menuClassName="annotation-name-combobox__menu annotation-name-combobox__menu--portal"
    />
  );
}
